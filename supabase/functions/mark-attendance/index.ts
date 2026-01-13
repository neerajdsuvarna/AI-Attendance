import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Max-Age': '86400',
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { 
      status: 200,
      headers: corsHeaders
    })
  }

  try {
    // Get the authorization header
    const authHeader = req.headers.get('Authorization')
    if (!authHeader) {
      return new Response(
        JSON.stringify({ error: 'Missing authorization header' }),
        { 
          status: 401, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Create Supabase client with service role key for admin access
    const supabaseAdminClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // Create a client with user's token for verification
    const supabaseUserClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      {
        global: {
          headers: { Authorization: authHeader },
        },
      }
    )

    // Get the user from the auth token
    const {
      data: { user },
      error: userError,
    } = await supabaseUserClient.auth.getUser()

    if (userError || !user) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { 
          status: 401, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Check user role - only super_admin and hr_admin can mark attendance
    const { data: profile, error: profileError } = await supabaseAdminClient
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single()

    if (profileError || !profile) {
      return new Response(
        JSON.stringify({ error: 'Failed to fetch user profile' }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    const userRole = profile.role
    if (userRole !== 'super_admin' && userRole !== 'hr_admin') {
      return new Response(
        JSON.stringify({ error: 'Access denied. Only administrators can mark attendance.' }),
        { 
          status: 403, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Parse request body
    const body = await req.json()
    const { employee_id, action } = body

    if (!employee_id || !action) {
      return new Response(
        JSON.stringify({ error: 'Missing required fields: employee_id and action' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    if (action !== 'entry' && action !== 'exit') {
      return new Response(
        JSON.stringify({ error: 'Invalid action. Must be "entry" or "exit"' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Verify employee exists
    const { data: employee, error: employeeError } = await supabaseAdminClient
      .from('employees')
      .select('id, name')
      .eq('id', employee_id)
      .single()

    if (employeeError || !employee) {
      return new Response(
        JSON.stringify({ error: 'Employee not found' }),
        { 
          status: 404, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    const now = new Date().toISOString()

    if (action === 'entry') {
      // Check if employee already has an open entry (no exit_time) for today
      const today = new Date()
      today.setHours(0, 0, 0, 0)
      const todayStart = today.toISOString()
      const todayEnd = new Date(today)
      todayEnd.setHours(23, 59, 59, 999)
      const todayEndISO = todayEnd.toISOString()

      const { data: existingEntry, error: checkError } = await supabaseAdminClient
        .from('attendance')
        .select('id, entry_time, exit_time')
        .eq('employee_id', employee_id)
        .gte('entry_time', todayStart)
        .lte('entry_time', todayEndISO)
        .is('exit_time', null)
        .order('entry_time', { ascending: false })
        .limit(1)
        .single()

      // If there's already an open entry for today, don't create a new one
      if (existingEntry && !existingEntry.exit_time) {
        return new Response(
          JSON.stringify({
            success: true,
            message: 'Entry already exists for today',
            action: 'entry',
            attendance_id: existingEntry.id,
            entry_time: existingEntry.entry_time,
            employee_name: employee.name
          }),
          { 
            status: 200, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      // Check if there was a recent exit (within last 5 minutes) to prevent rapid re-entries
      const { data: recentExit, error: recentExitError } = await supabaseAdminClient
        .from('attendance')
        .select('id, exit_time')
        .eq('employee_id', employee_id)
        .not('exit_time', 'is', null)
        .order('exit_time', { ascending: false })
        .limit(1)
        .single()

      if (recentExit && recentExit.exit_time) {
        const exitTime = new Date(recentExit.exit_time)
        const nowTime = new Date(now)
        const minutesSinceExit = (nowTime.getTime() - exitTime.getTime()) / (1000 * 60)
        
        // If exit was less than 5 minutes ago, don't create a new entry (likely a false exit)
        if (minutesSinceExit < 5) {
          return new Response(
            JSON.stringify({
              success: false,
              message: `Cannot create new entry. Last exit was ${Math.round(minutesSinceExit)} minutes ago. Please wait at least 5 minutes between entries.`,
              action: 'entry',
              employee_name: employee.name
            }),
            { 
              status: 200, 
              headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
            }
          )
        }
      }

      // Create new entry
      const { data: attendance, error: insertError } = await supabaseAdminClient
        .from('attendance')
        .insert({
          employee_id: employee_id,
          entry_time: now
        })
        .select()
        .single()

      if (insertError) {
        console.error('Error inserting attendance entry:', insertError)
        return new Response(
          JSON.stringify({ 
            error: 'Failed to mark entry',
            details: insertError.message 
          }),
          { 
            status: 500, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      return new Response(
        JSON.stringify({
          success: true,
          message: `Entry marked for ${employee.name}`,
          action: 'entry',
          attendance_id: attendance.id,
          entry_time: attendance.entry_time,
          employee_name: employee.name
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      )

    } else if (action === 'exit') {
      // Find the most recent entry without an exit_time for today
      const today = new Date()
      today.setHours(0, 0, 0, 0)
      const todayStart = today.toISOString()
      const todayEnd = new Date(today)
      todayEnd.setHours(23, 59, 59, 999)
      const todayEndISO = todayEnd.toISOString()

      const { data: openEntry, error: findError } = await supabaseAdminClient
        .from('attendance')
        .select('id, entry_time')
        .eq('employee_id', employee_id)
        .gte('entry_time', todayStart)
        .lte('entry_time', todayEndISO)
        .is('exit_time', null)
        .order('entry_time', { ascending: false })
        .limit(1)
        .single()

      if (findError || !openEntry) {
        return new Response(
          JSON.stringify({ 
            error: 'No open entry found to mark exit',
            details: 'Employee must have an entry before marking exit'
          }),
          { 
            status: 400, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      // Update the entry with exit time
      const { data: updatedAttendance, error: updateError } = await supabaseAdminClient
        .from('attendance')
        .update({ exit_time: now })
        .eq('id', openEntry.id)
        .select()
        .single()

      if (updateError) {
        console.error('Error updating attendance exit:', updateError)
        return new Response(
          JSON.stringify({ 
            error: 'Failed to mark exit',
            details: updateError.message 
          }),
          { 
            status: 500, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      return new Response(
        JSON.stringify({
          success: true,
          message: `Exit marked for ${employee.name}`,
          action: 'exit',
          attendance_id: updatedAttendance.id,
          entry_time: updatedAttendance.entry_time,
          exit_time: updatedAttendance.exit_time,
          employee_name: employee.name
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      )
    }

  } catch (error) {
    console.error('Error in mark-attendance function:', error)
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        details: error.message 
      }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )
  }
})

