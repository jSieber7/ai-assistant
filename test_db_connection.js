const { Pool } = require('pg');

const pool = new Pool({
  user: 'postgres',
  host: 'supabase-db',
  database: 'postgres',
  password: 'dev-password',
  port: 5432,
});

async function testConnection() {
  try {
    const client = await pool.connect();
    console.log('Connected to database successfully');
    const res = await client.query('SELECT NOW()');
    console.log('Current time:', res.rows[0].now);
    client.release();
    process.exit(0);
  } catch (err) {
    console.error('Error connecting to database:', err);
    process.exit(1);
  }
}

testConnection();
