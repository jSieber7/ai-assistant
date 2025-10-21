#!/usr/bin/env python3
"""
Helper script to generate a secure JWT secret and corresponding Supabase API keys.
Run this script to generate new secrets for your self-hosted Supabase instance.
"""

import secrets
import jwt
import base64
import sys

def generate_jwt_secret():
    """Generate a secure, 64-character JWT secret."""
    return secrets.token_urlsafe(48)

def generate_supabase_keys(jwt_secret):
    """
    Generate Supabase anon and service_role keys from a JWT secret.
    
    Args:
        jwt_secret (str): The JWT secret to use for signing the tokens.
        
    Returns:
        tuple: (anon_key, service_role_key)
    """
    # Payload for anon key
    anon_payload = {
        "role": "anon",
        "iss": "supabase",
        "iat": 1600000000,  # Fixed timestamp for consistency
        "exp": 4600000000   # Far future expiration
    }
    
    # Payload for service_role key
    service_role_payload = {
        "role": "service_role",
        "iss": "supabase",
        "iat": 1600000000,
        "exp": 4600000000
    }
    
    # Encode the JWTs
    anon_key = jwt.encode(anon_payload, jwt_secret, algorithm="HS256")
    service_role_key = jwt.encode(service_role_payload, jwt_secret, algorithm="HS256")
    
    return anon_key, service_role_key

def main():
    """Main function to generate and print the secrets."""
    print("🔐 Generating new Supabase secrets...")
    
    # Generate a new JWT secret
    jwt_secret = generate_jwt_secret()
    print(f"\nGenerated JWT_SECRET:\n{jwt_secret}")
    
    # Generate the API keys
    anon_key, service_role_key = generate_supabase_keys(jwt_secret)
    
    print(f"\nGenerated SUPABASE_ANON_KEY:\n{anon_key}")
    print(f"\nGenerated SUPABASE_SERVICE_ROLE_KEY:\n{service_role_key}")
    
    print("\n" + "="*60)
    print("📋 Instructions:")
    print("1. Copy these values into your .env.dev file.")
    print("2. Ensure your docker-compose.yml references these variables.")
    print("3. Restart your Docker services: docker compose --profile dev down && docker compose --profile dev up -d")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: Missing required library. Please install it with: pip install {e.name}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)