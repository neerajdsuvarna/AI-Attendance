import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Get the authorization header - check multiple possible header names
    const authHeader = req.headers.get('Authorization') || 
                      req.headers.get('authorization') ||
                      req.headers.get('x-authorization')
    
    console.log('Auth header present:', !!authHeader)
    console.log('Request headers:', Object.fromEntries(req.headers.entries()))
    
    if (!authHeader) {
      console.error('Missing authorization header')
      return new Response(
        JSON.stringify({ error: 'Missing authorization header' }),
        { 
          status: 401, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Get environment variables
    const supabaseUrl = Deno.env.get('SUPABASE_URL')
    const supabaseAnonKey = Deno.env.get('SUPABASE_ANON_KEY')
    
    if (!supabaseUrl || !supabaseAnonKey) {
      console.error('Missing Supabase environment variables')
      return new Response(
        JSON.stringify({ error: 'Server configuration error' }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Create Supabase client with ANON_KEY for user verification (like interviewcoach)
    const supabaseClient = createClient(
      supabaseUrl,
      supabaseAnonKey,
      {
        global: {
          headers: { Authorization: authHeader },
        },
      }
    )

    // Verify user authentication with ANON_KEY client
    const {
      data: { user },
      error: userError,
    } = await supabaseClient.auth.getUser()

    console.log('User verification result:', { 
      hasUser: !!user, 
      userId: user?.id, 
      error: userError?.message 
    })

    if (userError || !user) {
      console.error('User verification failed:', userError)
      return new Response(
        JSON.stringify({ 
          error: 'Unauthorized', 
          message: userError?.message || 'Invalid or missing auth token' 
        }),
        { 
          status: 401, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Get user role from profiles table using the user client (with RLS)
    const { data: profile, error: profileError } = await supabaseClient
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single()

    if (profileError) {
      console.error('Error fetching profile:', profileError)
      return new Response(
        JSON.stringify({ error: 'Failed to fetch user role' }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    const role = profile?.role || null

    // Determine dashboard route based on role
    let dashboardRoute = '/dashboard'
    const allowedRoles = ['super_admin', 'hr_admin', 'employee']
    
    if (role && allowedRoles.includes(role)) {
      // All roles go to same dashboard for now, but you can customize:
      // dashboardRoute = role === 'super_admin' ? '/admin-dashboard' : '/dashboard'
      dashboardRoute = '/dashboard'
    } else {
      return new Response(
        JSON.stringify({ 
          error: 'Invalid user role',
          role: role 
        }),
        { 
          status: 403, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    return new Response(
      JSON.stringify({
        success: true,
        userId: user.id,
        role: role,
        dashboardRoute: dashboardRoute,
        email: user.email,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  } catch (error) {
    console.error('Error in check-user-role function:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )
  }
})

