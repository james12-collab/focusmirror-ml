import os 
from urllib.parse import urlsplit 
from supabase import create_client 
 
def get_supabase_client(url=None, key=None): 
    u = url or os.environ.get('SUPABASE_URL', '') 
    k = key or os.environ.get('SUPABASE_KEY', '') 
    parts = urlsplit(u) 
    if parts.scheme and parts.netloc: 
        u = parts.scheme + '://' + parts.netloc 
    if not u or not k: 
        raise ValueError('Missing Supabase credentials.') 
    return create_client(u, k) 
 
def fetch_all_sessions(client): 
    res = client.table('sessions').select('*').execute() 
    return res.data 
