// MongoDB initialization script for multi-writer system
// This script creates the database and initial collections with indexes

// Switch to the multi-writer database
db = db.getSiblingDB('multi_writer_system');

// Create collections
print('Creating collections...');

// Workflows collection
db.createCollection('workflows');
print('Created workflows collection');

// Content collection
db.createCollection('content');
print('Created content collection');

// Check results collection
db.createCollection('check_results');
print('Created check_results collection');

// Create indexes for better performance
print('Creating indexes...');

// Workflows collection indexes
db.workflows.createIndex({ "workflow_id": 1 }, { unique: true });
db.workflows.createIndex({ "status": 1 });
db.workflows.createIndex({ "created_at": 1 });
db.workflows.createIndex({ "updated_at": 1 });
print('Created indexes for workflows collection');

// Content collection indexes
db.content.createIndex({ "workflow_id": 1 });
db.content.createIndex({ "writer_id": 1 });
db.content.createIndex({ "created_at": 1 });
print('Created indexes for content collection');

// Check results collection indexes
db.check_results.createIndex({ "workflow_id": 1 });
db.check_results.createIndex({ "checker_id": 1 });
db.check_results.createIndex({ "created_at": 1 });
print('Created indexes for check_results collection');

// Create a user for the application (optional)
print('Creating application user...');
db.createUser({
  user: 'multi_writer_app',
  pwd: 'app_password123',
  roles: [
    {
      role: 'readWrite',
      db: 'multi_writer_system'
    }
  ]
});
print('Created application user');

// Insert some sample data for testing (optional)
print('Inserting sample data...');

// Sample workflow
db.workflows.insertOne({
  workflow_id: 'sample_workflow_001',
  prompt: 'Sample prompt for testing',
  sources: [{ url: 'https://example.com/sample' }],
  status: 'completed',
  stages: {
    source_processing: { metadata: { total_sources: 1, successful: 1, failed: 0 } },
    content_generation: { total_versions: 3 },
    quality_checking: { best_score: 85.5, passes_threshold: true },
    template_rendering: { template_used: 'article.html.jinja' }
  },
  final_output: '<h1>Sample Content</h1><p>This is sample content for testing.</p>',
  errors: [],
  created_at: new Date(),
  updated_at: new Date()
});

// Sample content
db.content.insertOne({
  workflow_id: 'sample_workflow_001',
  writer_id: 'technical_1',
  specialty: 'technical',
  content: 'This is sample technical content generated for testing purposes.',
  model_used: 'claude-3.5-sonnet',
  sources_used: ['https://example.com/sample'],
  word_count: 12,
  confidence_score: 0.85,
  created_at: new Date()
});

// Sample check result
db.check_results.insertOne({
  workflow_id: 'sample_workflow_001',
  checker_id: 'factual_1',
  focus_area: 'factual',
  original_content: 'This is sample technical content generated for testing purposes.',
  score: 85,
  issues_found: [],
  improvements: [
    { type: 'clarity', description: 'Added more specific examples' }
  ],
  improved_content: 'This is sample technical content with specific examples for testing purposes.',
  recommendations: ['Consider adding more technical details'],
  created_at: new Date()
});

print('Sample data inserted successfully');
print('MongoDB initialization completed successfully!');