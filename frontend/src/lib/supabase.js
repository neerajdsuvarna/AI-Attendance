import { createClient } from '@supabase/supabase-js'

// Replace these with your Supabase project URL and anon key
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'http://127.0.0.1:54321'
// Default anon key for local Supabase development
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0'

// Clear old Supabase storage if URL has changed
// Supabase stores auth tokens with keys like: sb-<project-ref>-auth-token
// Extract project ref from URL (e.g., homvsidsakohneaaxcfz from https://homvsidsakohneaaxcfz.supabase.co)
const currentProjectRef = supabaseUrl.match(/https?:\/\/([^.]+)\.supabase\.co/)?.[1]

if (currentProjectRef && typeof window !== 'undefined') {
  // Find all Supabase storage keys
  const allStorageKeys = Object.keys(localStorage)
  const supabaseStorageKeys = allStorageKeys.filter(key => 
    key.startsWith('sb-') && key.endsWith('-auth-token')
  )
  
  // Clear storage keys that don't match the current project
  supabaseStorageKeys.forEach(key => {
    const keyProjectRef = key.match(/sb-([^-]+)-auth-token/)?.[1]
    if (keyProjectRef && keyProjectRef !== currentProjectRef) {
      console.log(`Clearing old Supabase storage for project: ${keyProjectRef}`)
      localStorage.removeItem(key)
      // Also clear any related storage
      Object.keys(localStorage).forEach(k => {
        if (k.includes(keyProjectRef)) {
          localStorage.removeItem(k)
        }
      })
    }
  })
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

