#!/bin/bash

# =============================================================================
# Fix .localhost Domains for Traefik
# =============================================================================
# This script adds .localhost domains to /etc/hosts file so they resolve properly
# Usage: sudo bash utility/fix_localhost_domains.sh

echo "Adding .localhost domains to /etc/hosts file..."

# Backup the original hosts file
sudo cp /etc/hosts /etc/hosts.backup

# Add .localhost domains to hosts file
{
    echo "127.0.0.1 frontend.localhost"
    echo "127.0.0.1 firecrawl.localhost"
    echo "127.0.0.1 searxng.localhost"
    echo "127.0.0.1 supabase.localhost"
    echo "127.0.0.1 traefik.localhost"
} | sudo tee -a /etc/hosts

echo "✅ .localhost domains added to /etc/hosts"
echo ""
echo "You can now access services at:"
echo "  Frontend: http://frontend.localhost:8880"
echo "  Firecrawl API: http://firecrawl.localhost:8880"
echo "  SearXNG: http://searxng.localhost:8880"
echo "  Supabase: http://supabase.localhost:8880"
echo "  Traefik Dashboard: http://traefik.localhost:8881/dashboard/"
echo ""
echo "To undo these changes: sudo cp /etc/hosts.backup /etc/hosts"