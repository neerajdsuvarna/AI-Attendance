import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
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

    // Check user role
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
    const isAdmin = userRole === 'super_admin' || userRole === 'hr_admin'

    // Parse query parameters
    const url = new URL(req.url)
    const startDate = url.searchParams.get('start_date')
    const endDate = url.searchParams.get('end_date')
    const employeeId = url.searchParams.get('employee_id')
    const date = url.searchParams.get('date') // Single date filter
    const status = url.searchParams.get('status') // 'present', 'exited', 'all'

    // Build query
    let query = supabaseAdminClient
      .from('attendance')
      .select(`
        id,
        employee_id,
        entry_time,
        exit_time,
        created_at,
        updated_at,
        employees (
          id,
          name,
          email
        )
      `)
      .order('entry_time', { ascending: false })

    // If employee (not admin), only show their own attendance
    if (!isAdmin) {
      // Get employee_id from employees table where profile_id matches user.id
      const { data: employeeData } = await supabaseAdminClient
        .from('employees')
        .select('id')
        .eq('profile_id', user.id)
        .single()

      if (employeeData) {
        query = query.eq('employee_id', employeeData.id)
      } else {
        // Employee not found in employees table, return empty
        return new Response(
          JSON.stringify({
            success: true,
            attendance: [],
            summary: {
              total_records: 0,
              present_today: 0,
              exited_today: 0
            }
          }),
          {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        )
      }
    } else if (employeeId) {
      // Admin filtering by specific employee
      query = query.eq('employee_id', employeeId)
    }

    // Date filtering
    if (date) {
      // Single date filter
      const dateStart = new Date(date)
      dateStart.setHours(0, 0, 0, 0)
      const dateEnd = new Date(date)
      dateEnd.setHours(23, 59, 59, 999)
      
      query = query
        .gte('entry_time', dateStart.toISOString())
        .lte('entry_time', dateEnd.toISOString())
    } else if (startDate && endDate) {
      // Date range filter
      const start = new Date(startDate)
      start.setHours(0, 0, 0, 0)
      const end = new Date(endDate)
      end.setHours(23, 59, 59, 999)
      
      query = query
        .gte('entry_time', start.toISOString())
        .lte('entry_time', end.toISOString())
    }

    // Status filtering
    if (status === 'present') {
      query = query.is('exit_time', null)
    } else if (status === 'exited') {
      query = query.not('exit_time', 'is', null)
    }
    // 'all' or no status filter means show all

    // Execute query
    const { data: attendance, error: attendanceError } = await query

    if (attendanceError) {
      console.error('Error fetching attendance:', attendanceError)
      return new Response(
        JSON.stringify({ 
          error: 'Failed to fetch attendance',
          details: attendanceError.message 
        }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Calculate summary statistics
    const today = new Date()
    today.setHours(0, 0, 0, 0)
    const todayEnd = new Date(today)
    todayEnd.setHours(23, 59, 59, 999)

    const todayRecords = attendance?.filter(record => {
      const entryDate = new Date(record.entry_time)
      return entryDate >= today && entryDate <= todayEnd
    }) || []

    const presentToday = todayRecords.filter(r => r.exit_time === null).length
    const exitedToday = todayRecords.filter(r => r.exit_time !== null).length

    // Format response
    const formattedAttendance = attendance?.map(record => ({
      id: record.id,
      employee_id: record.employee_id,
      employee_name: record.employees?.name || 'Unknown',
      employee_email: record.employees?.email || '',
      entry_time: record.entry_time,
      exit_time: record.exit_time,
      created_at: record.created_at,
      updated_at: record.updated_at,
      status: record.exit_time === null ? 'present' : 'exited',
      hours_worked: record.exit_time 
        ? ((new Date(record.exit_time).getTime() - new Date(record.entry_time).getTime()) / (1000 * 60 * 60)).toFixed(2)
        : null
    })) || []

    return new Response(
      JSON.stringify({
        success: true,
        attendance: formattedAttendance,
        summary: {
          total_records: formattedAttendance.length,
          present_today: presentToday,
          exited_today: exitedToday,
          total_today: todayRecords.length
        }
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )

  } catch (error) {
    console.error('Error in get-attendance function:', error)
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

