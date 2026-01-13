import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
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
    const { data: { user }, error: userError } = await supabaseUserClient.auth.getUser()

    if (userError || !user) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        {
          status: 401,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    // Check user role - allow all authenticated users (employees can see their own, admins see all)
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

    // Fetch all employees
    const { data: employees, error: employeesError } = await supabaseAdminClient
      .from('employees')
      .select('id, name, email, profile_id, face_embeddings, created_at, updated_at')
      .order('created_at', { ascending: false })

    if (employeesError) {
      console.error('Error fetching employees:', employeesError)
      return new Response(
        JSON.stringify({
          error: 'Failed to fetch employees',
          details: employeesError.message
        }),
        {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      )
    }

    // Fetch profiles for all employees to get their roles
    const profileIds = employees.map(emp => emp.profile_id).filter(Boolean)
    const { data: profiles, error: profilesError } = profileIds.length > 0
      ? await supabaseAdminClient
          .from('profiles')
          .select('id, role')
          .in('id', profileIds)
      : { data: [], error: null }

    if (profilesError) {
      console.error('Error fetching profiles:', profilesError)
      // Continue without profiles - role will be null
    }

    // Create a map of profile_id to role
    const profileRoleMap = new Map()
    if (profiles) {
      profiles.forEach(profile => {
        profileRoleMap.set(profile.id, profile.role)
      })
    }

    // Separate HR admins and employees
    // HR Admins: Only those with profile_id pointing to a profile with role='hr_admin' or role='super_admin'
    // Employees: Everyone else (no profile_id, or profile with any other role, or null role)
    const hrAdmins: any[] = []
    const regularEmployees: any[] = []

    employees.forEach((emp: any) => {
      // Get role from profile if profile_id exists
      const role = emp.profile_id ? profileRoleMap.get(emp.profile_id) || null : null
      
      const employeeData = {
        id: emp.id,
        name: emp.name,
        email: emp.email,
        profile_id: emp.profile_id,
        role: role,
        has_face_embeddings: !!emp.face_embeddings,
        created_at: emp.created_at,
        updated_at: emp.updated_at
      }

      // Only classify as HR Admin if profile role is explicitly 'hr_admin' or 'super_admin'
      // Everything else (null, undefined, 'employee', or any other value) goes to regular employees
      if (role === 'hr_admin' || role === 'super_admin') {
        hrAdmins.push(employeeData)
      } else {
        // All others are regular employees (including those with no profile_id or any other role)
        regularEmployees.push(employeeData)
      }
    })

    return new Response(
      JSON.stringify({
        success: true,
        hr_admins: hrAdmins,
        employees: regularEmployees,
        total_hr_admins: hrAdmins.length,
        total_employees: regularEmployees.length,
        total: employees.length
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )

  } catch (error) {
    console.error('Error in get-employees-list function:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    )
  }
})

