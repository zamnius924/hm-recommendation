# %% 
import duckdb
from scripts.load_db import load_db

# %% Connect to DuckDB
con = load_db()
con.execute("SHOW TABLES").df()
