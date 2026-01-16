import { supabase } from './supabase'

/**
 * Helper function to call Supabase Edge Functions with proper authentication
 * This ensures consistent authentication handling across the app
 */
export async function callEdgeFunction(functionName, options = {}) {
  try {
    // Get current session
    const { data: { session }, error: sessionError } = await supabase.auth.getSession()
    
    if (sessionError || !session) {
      throw new Error('Not authenticated. Please log in again.')
    }

    // Build URL with query parameters if provided
    const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'http://127.0.0.1:54321'
    let url = `${supabaseUrl}/functions/v1/${functionName}`
    
    // Add query parameters if provided
    if (options.queryParams) {
      const params = new URLSearchParams(options.queryParams)
      const queryString = params.toString()
      if (queryString) {
        url += `?${queryString}`
      }
    }

    // Prepare headers - only Authorization is needed, NOT apikey
    const headers = {
      'Authorization': `Bearer ${session.access_token}`,
      'Content-Type': 'application/json',
    }

    // Add any additional headers
    if (options.headers) {
      Object.assign(headers, options.headers)
    }

    // Make the request
    const response = await fetch(url, {
      method: options.method || 'GET',
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    })

    // Handle response
    if (!response.ok) {
      let errorMessage = `HTTP error! status: ${response.status}`
      try {
        const errorData = await response.json()
        errorMessage = errorData.error || errorData.message || errorMessage
      } catch (e) {
        // If response is not JSON, try to get text
        try {
          const errorText = await response.text()
          errorMessage = errorText || errorMessage
        } catch (e2) {
          // Use default error message
        }
      }
      throw new Error(errorMessage)
    }

    // Parse and return JSON response
    try {
      return await response.json()
    } catch (e) {
      // If response is not JSON, return text
      return { data: await response.text() }
    }
  } catch (error) {
    console.error(`Error calling edge function ${functionName}:`, error)
    throw error
  }
}

/**
 * Convenience function for GET requests to edge functions
 */
export async function getEdgeFunction(functionName, queryParams = {}) {
  return callEdgeFunction(functionName, {
    method: 'GET',
    queryParams,
  })
}

/**
 * Convenience function for POST requests to edge functions
 */
export async function postEdgeFunction(functionName, body = {}, queryParams = {}) {
  return callEdgeFunction(functionName, {
    method: 'POST',
    body,
    queryParams,
  })
}

/**
 * Convenience function for PUT requests to edge functions
 */
export async function putEdgeFunction(functionName, body = {}, queryParams = {}) {
  return callEdgeFunction(functionName, {
    method: 'PUT',
    body,
    queryParams,
  })
}

/**
 * Convenience function for DELETE requests to edge functions
 */
export async function deleteEdgeFunction(functionName, queryParams = {}) {
  return callEdgeFunction(functionName, {
    method: 'DELETE',
    queryParams,
  })
}
