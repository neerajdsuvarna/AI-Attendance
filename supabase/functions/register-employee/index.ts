import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Max-Age': '86400',
}

serve(async (req) => {
  // Handle CORS preflight requests - must be first
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
    // Note: We use service role key for admin operations, but verify user separately
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

    // Get the user from the auth token using user client
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

    // Check user role - only super_admin can register all types, hr_admin can only register employees
    const { data: profile, error: profileError } = await supabaseUserClient
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single()

    if (profileError || !profile) {
      return new Response(
        JSON.stringify({ error: 'Failed to fetch user role' }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    const userRole = profile.role
    if (userRole !== 'super_admin' && userRole !== 'hr_admin') {
      return new Response(
        JSON.stringify({ error: 'Insufficient permissions. Only admins can register users.' }),
        { 
          status: 403, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Parse request body
    const body = await req.json()
    const { name, email, role, face_embeddings } = body

    // Validate required fields
    if (!name || !email) {
      return new Response(
        JSON.stringify({ error: 'Name and email are required' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      return new Response(
        JSON.stringify({ error: 'Invalid email format' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Validate and set role
    const allowedRoles = ['employee', 'hr_admin', 'super_admin']
    const targetRole = role || 'employee'
    
    // HR admins can only register employees
    if (userRole === 'hr_admin' && targetRole !== 'employee') {
      return new Response(
        JSON.stringify({ error: 'HR admins can only register employees' }),
        { 
          status: 403, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Only super_admin can create hr_admin or super_admin
    if ((targetRole === 'hr_admin' || targetRole === 'super_admin') && userRole !== 'super_admin') {
      return new Response(
        JSON.stringify({ error: 'Only super admins can create admin users' }),
        { 
          status: 403, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    if (!allowedRoles.includes(targetRole)) {
      return new Response(
        JSON.stringify({ error: 'Invalid role. Allowed roles: employee, hr_admin, super_admin' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    const trimmedEmail = email.trim().toLowerCase()
    const trimmedName = name.trim()

    // Check if email exists in employees table
    const { data: existingEmployee, error: checkEmployeeError } = await supabaseAdminClient
      .from('employees')
      .select('id, email')
      .eq('email', trimmedEmail)
      .maybeSingle()

    if (checkEmployeeError && checkEmployeeError.code !== 'PGRST116') {
      console.error('Error checking existing employee:', checkEmployeeError)
      return new Response(
        JSON.stringify({ error: 'Failed to check existing employee' }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    if (existingEmployee) {
      return new Response(
        JSON.stringify({ error: 'User with this email already exists' }),
        { 
          status: 409, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // For admin users, also check if email exists in auth.users
    if (targetRole === 'hr_admin' || targetRole === 'super_admin') {
      try {
        const { data: authUsers } = await supabaseAdminClient.auth.admin.listUsers()
        const emailExists = authUsers?.users?.some(u => u.email === trimmedEmail)
        
        if (emailExists) {
          return new Response(
            JSON.stringify({ error: 'User with this email already exists' }),
            { 
              status: 409, 
              headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
            }
          )
        }
      } catch (error) {
        console.error('Error checking auth users:', error)
        // Continue - will fail at invite step if email exists
      }
    }

    // Get the profile_id (current user's profile)
    const { data: userProfile, error: userProfileError } = await supabaseAdminClient
      .from('profiles')
      .select('id')
      .eq('id', user.id)
      .single()

    if (userProfileError || !userProfile) {
      return new Response(
        JSON.stringify({ error: 'Failed to fetch user profile' }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    let authUserId: string | null = null
    let profileId: string | null = null
    let employeeId: string | null = null

    // Handle different registration scenarios
    if (targetRole === 'employee') {
      // Employee: Only register in employees table (NO profile_id link)
      const { data: newEmployee, error: insertError } = await supabaseAdminClient
        .from('employees')
        .insert({
          name: trimmedName,
          email: trimmedEmail,
          profile_id: null, // Employees don't have profile_id linked
          face_embeddings: face_embeddings || null,
        })
        .select()
        .single()

      if (insertError) {
        console.error('Error inserting employee:', insertError)
        return new Response(
          JSON.stringify({ 
            error: 'Failed to register employee',
            details: insertError.message 
          }),
          { 
            status: 500, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      employeeId = newEmployee.id

      return new Response(
        JSON.stringify({
          success: true,
          message: 'Employee registered successfully',
          user: {
            id: newEmployee.id,
            name: newEmployee.name,
            email: newEmployee.email,
            role: 'employee',
            has_face_embeddings: !!newEmployee.face_embeddings,
            requires_login: false,
          },
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      )

    } else if (targetRole === 'hr_admin') {
      // HR Admin: Create auth user, profile, AND employee record with profile_id linked
      
      // 1. Create auth user and send invite
      const { data: authUser, error: authError } = await supabaseAdminClient.auth.admin.inviteUserByEmail(
        trimmedEmail,
        {
          data: {
            name: trimmedName,
            role: 'hr_admin',
          },
          redirectTo: `${Deno.env.get('SITE_URL') || 'http://localhost:3000'}/auth/callback?type=invite`,
        }
      )

      if (authError || !authUser?.user) {
        console.error('Error creating auth user:', authError)
        return new Response(
          JSON.stringify({ 
            error: 'Failed to create user account',
            details: authError?.message 
          }),
          { 
            status: 500, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      authUserId = authUser.user.id

      // 2. Create profile (trigger should handle this, but we'll ensure it exists)
      const { data: newProfile, error: profileInsertError } = await supabaseAdminClient
        .from('profiles')
        .upsert({
          id: authUserId,
          role: 'hr_admin',
        }, {
          onConflict: 'id',
        })
        .select()
        .single()

      if (profileInsertError) {
        console.error('Error creating profile:', profileInsertError)
        // Try to continue - profile might have been created by trigger
      } else {
        profileId = newProfile.id
      }

      // 3. Create employee record with profile_id linked (only HR admins get profile_id)
      const { data: newEmployee, error: employeeError } = await supabaseAdminClient
        .from('employees')
        .insert({
          name: trimmedName,
          email: trimmedEmail,
          profile_id: authUserId, // HR admins: Link to their own profile
          face_embeddings: face_embeddings || null,
        })
        .select()
        .single()

      if (employeeError) {
        console.error('Error creating employee record:', employeeError)
        // Clean up: delete auth user if employee creation fails
        await supabaseAdminClient.auth.admin.deleteUser(authUserId)
        return new Response(
          JSON.stringify({ 
            error: 'Failed to create employee record',
            details: employeeError.message 
          }),
          { 
            status: 500, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      employeeId = newEmployee.id

      return new Response(
        JSON.stringify({
          success: true,
          message: 'HR Admin registered successfully. Invitation email sent.',
          user: {
            id: newEmployee.id,
            name: newEmployee.name,
            email: newEmployee.email,
            role: 'hr_admin',
            has_face_embeddings: !!newEmployee.face_embeddings,
            requires_login: true,
            invitation_sent: true,
          },
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      )

    } else if (targetRole === 'super_admin') {
      // Super Admin: Create auth user and profile only (NOT in employees table)
      
      // 1. Create auth user and send invite
      const { data: authUser, error: authError } = await supabaseAdminClient.auth.admin.inviteUserByEmail(
        trimmedEmail,
        {
          data: {
            name: trimmedName,
            role: 'super_admin',
          },
          redirectTo: `${Deno.env.get('SITE_URL') || 'http://localhost:3000'}/auth/callback?type=invite`,
        }
      )

      if (authError || !authUser?.user) {
        console.error('Error creating auth user:', authError)
        return new Response(
          JSON.stringify({ 
            error: 'Failed to create user account',
            details: authError?.message 
          }),
          { 
            status: 500, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        )
      }

      authUserId = authUser.user.id

      // 2. Create profile
      const { data: newProfile, error: profileInsertError } = await supabaseAdminClient
        .from('profiles')
        .upsert({
          id: authUserId,
          role: 'super_admin',
        }, {
          onConflict: 'id',
        })
        .select()
        .single()

      if (profileInsertError) {
        console.error('Error creating profile:', profileInsertError)
        // Try to continue - profile might have been created by trigger
      } else {
        profileId = newProfile.id
      }

      return new Response(
        JSON.stringify({
          success: true,
          message: 'Super Admin registered successfully. Invitation email sent.',
          user: {
            id: authUserId,
            name: trimmedName,
            email: trimmedEmail,
            role: 'super_admin',
            requires_login: true,
            invitation_sent: true,
          },
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      )
    }

    // Should not reach here
    return new Response(
      JSON.stringify({ error: 'Invalid role processing' }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )

  } catch (error) {
    console.error('Error in register-employee function:', error)
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        message: error.message 
      }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )
  }
})
