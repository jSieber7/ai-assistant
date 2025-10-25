import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { 
  Bot, 
  Plus, 
  Save, 
  Play, 
  Trash2, 
  Download, 
  Upload,
  CheckCircle,
  AlertCircle,
  Info,
  Code,
  Settings,
  Zap
} from 'lucide-react';
import { 
  apiService, 
  AgentDesignRequest, 
  AgentDesignResponse, 
  AgentValidationResponse,
  SavedAgent,
  AgentDesignerTool
} from '@/services/api';
import { showToast } from '@/lib/toast';

interface AgentDesignerProps {
  onAgentCreated?: (agent: AgentDesignResponse) => void;
}

const AgentDesigner: React.FC<AgentDesignerProps> = ({ onAgentCreated }) => {
  // Form state
  const [agentName, setAgentName] = useState('');
  const [agentDescription, setAgentDescription] = useState('');
  const [agentRequirements, setAgentRequirements] = useState('');
  const [agentCategory, setAgentCategory] = useState('custom');
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [modelPreference, setModelPreference] = useState('');
  const [additionalContext, setAdditionalContext] = useState('');

  // UI state
  const [isCreating, setIsCreating] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [activeTab, setActiveTab] = useState('design');
  const [savedAgents, setSavedAgents] = useState<SavedAgent[]>([]);
  const [availableTools, setAvailableTools] = useState<AgentDesignerTool[]>([]);
  const [generatedCode, setGeneratedCode] = useState('');
  const [validationResult, setValidationResult] = useState<AgentValidationResponse | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<SavedAgent | null>(null);

  // Categories for agents
  const agentCategories = [
    { value: 'custom', label: 'Custom' },
    { value: 'content', label: 'Content' },
    { value: 'search', label: 'Search' },
    { value: 'data', label: 'Data' },
    { value: 'communication', label: 'Communication' },
    { value: 'utility', label: 'Utility' },
    { value: 'specialized', label: 'Specialized' }
  ];

  // Load saved agents and available tools on component mount
  useEffect(() => {
    loadSavedAgents();
    loadAvailableTools();
  }, []);

  const loadSavedAgents = async () => {
    try {
      const agents = await apiService.listSavedAgents();
      setSavedAgents(agents);
    } catch (error) {
      console.error('Failed to load saved agents:', error);
      showToast.error('Failed to load saved agents');
    }
  };

  const loadAvailableTools = async () => {
    try {
      const response = await apiService.getAvailableToolsForDesigner();
      setAvailableTools(response.tools);
    } catch (error) {
      console.error('Failed to load available tools:', error);
      showToast.error('Failed to load available tools');
    }
  };

  const handleToolToggle = (toolName: string) => {
    setSelectedTools(prev => 
      prev.includes(toolName) 
        ? prev.filter(t => t !== toolName)
        : [...prev, toolName]
    );
  };

  const validateForm = (): boolean => {
    if (!agentName.trim()) {
      showToast.error('Agent name is required');
      return false;
    }
    if (!agentDescription.trim()) {
      showToast.error('Agent description is required');
      return false;
    }
    if (!agentRequirements.trim()) {
      showToast.error('Agent requirements are required');
      return false;
    }
    return true;
  };

  const handleCreateAgent = async () => {
    if (!validateForm()) return;

    setIsCreating(true);
    try {
      const request: AgentDesignRequest = {
        name: agentName,
        description: agentDescription,
        requirements: agentRequirements,
        category: agentCategory,
        tools_needed: selectedTools,
        model_preference: modelPreference || undefined,
        additional_context: additionalContext || undefined
      };

      const response = await apiService.createAgent(request);
      
      if (response.success) {
        setGeneratedCode(response.agent_code || '');
        showToast.success(`Agent '${response.agent_name}' created successfully!`);
        onAgentCreated?.(response);
        loadSavedAgents(); // Refresh the list
        setActiveTab('code');
      } else {
        showToast.error(response.message);
      }
    } catch (error) {
      console.error('Failed to create agent:', error);
      showToast.error('Failed to create agent');
    } finally {
      setIsCreating(false);
    }
  };

  const handleValidateCode = async () => {
    if (!generatedCode.trim()) {
      showToast.error('No code to validate');
      return;
    }

    setIsValidating(true);
    try {
      const response = await apiService.validateAgent({
        agent_code: generatedCode,
        agent_name: agentName
      });
      setValidationResult(response);
      
      if (response.is_valid) {
        showToast.success('Agent code is valid!');
      } else {
        showToast.error('Agent code has validation errors');
      }
    } catch (error) {
      console.error('Failed to validate code:', error);
      showToast.error('Failed to validate agent code');
    } finally {
      setIsValidating(false);
    }
  };

  const handleDeleteAgent = async (agentId: string) => {
    if (!confirm('Are you sure you want to delete this agent?')) return;

    try {
      await apiService.deleteSavedAgent(agentId);
      showToast.success('Agent deleted successfully');
      loadSavedAgents(); // Refresh the list
      if (selectedAgent?.id === agentId) {
        setSelectedAgent(null);
        setGeneratedCode('');
      }
    } catch (error) {
      console.error('Failed to delete agent:', error);
      showToast.error('Failed to delete agent');
    }
  };

  const handleLoadAgent = async (agent: SavedAgent) => {
    try {
      const response = await apiService.getSavedAgent(agent.id);
      setGeneratedCode(response.agent_code);
      setSelectedAgent(agent);
      setActiveTab('code');
      showToast.success(`Loaded agent: ${agent.name}`);
    } catch (error) {
      console.error('Failed to load agent:', error);
      showToast.error('Failed to load agent');
    }
  };

  const handleActivateAgent = async (agentId: string) => {
    try {
      await apiService.activateSavedAgent(agentId);
      showToast.success('Agent activated successfully');
    } catch (error) {
      console.error('Failed to activate agent:', error);
      showToast.error('Failed to activate agent');
    }
  };

  const resetForm = () => {
    setAgentName('');
    setAgentDescription('');
    setAgentRequirements('');
    setAgentCategory('custom');
    setSelectedTools([]);
    setModelPreference('');
    setAdditionalContext('');
    setGeneratedCode('');
    setValidationResult(null);
    setSelectedAgent(null);
  };

  const downloadCode = () => {
    if (!generatedCode) return;
    
    const blob = new Blob([generatedCode], { type: 'text/python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${agentName.replace(/\s+/g, '_').toLowerCase()}_agent.py`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="flex items-center gap-3 mb-6">
        <Bot className="h-8 w-8 text-blue-600" />
        <h1 className="text-3xl font-bold">LLM Agent Designer</h1>
        <Badge variant="secondary">Beta</Badge>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="design" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Design
          </TabsTrigger>
          <TabsTrigger value="tools" className="flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Tools
          </TabsTrigger>
          <TabsTrigger value="code" className="flex items-center gap-2">
            <Code className="h-4 w-4" />
            Code
          </TabsTrigger>
          <TabsTrigger value="saved" className="flex items-center gap-2">
            <Save className="h-4 w-4" />
            Saved
          </TabsTrigger>
        </TabsList>

        <TabsContent value="design" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Agent Configuration</CardTitle>
              <CardDescription>
                Define the basic properties and requirements for your custom agent
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label htmlFor="agent-name" className="text-sm font-medium">
                    Agent Name *
                  </label>
                  <Input
                    id="agent-name"
                    placeholder="e.g., ContentAnalyzer, ResearchAssistant"
                    value={agentName}
                    onChange={(e) => setAgentName(e.target.value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <label htmlFor="agent-category" className="text-sm font-medium">
                    Category
                  </label>
                  <Select value={agentCategory} onValueChange={setAgentCategory}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a category" />
                    </SelectTrigger>
                    <SelectContent>
                      {agentCategories.map(category => (
                        <SelectItem key={category.value} value={category.value}>
                          {category.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <label htmlFor="agent-description" className="text-sm font-medium">
                  Description *
                </label>
                <Textarea
                  id="agent-description"
                  placeholder="Describe what this agent does and its main purpose..."
                  value={agentDescription}
                  onChange={(e) => setAgentDescription(e.target.value)}
                  rows={3}
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="agent-requirements" className="text-sm font-medium">
                  Requirements *
                </label>
                <Textarea
                  id="agent-requirements"
                  placeholder="Detailed requirements for the agent. What should it be able to do? How should it behave? What tools should it use?"
                  value={agentRequirements}
                  onChange={(e) => setAgentRequirements(e.target.value)}
                  rows={5}
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="model-preference" className="text-sm font-medium">
                  Model Preference (optional)
                </label>
                <Input
                  id="model-preference"
                  placeholder="e.g., gpt-4, claude-3, llama-3"
                  value={modelPreference}
                  onChange={(e) => setModelPreference(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="additional-context" className="text-sm font-medium">
                  Additional Context (optional)
                </label>
                <Textarea
                  id="additional-context"
                  placeholder="Any additional context, constraints, or special requirements..."
                  value={additionalContext}
                  onChange={(e) => setAdditionalContext(e.target.value)}
                  rows={3}
                />
              </div>

              <div className="flex gap-3 pt-4">
                <Button 
                  onClick={handleCreateAgent} 
                  disabled={isCreating}
                  className="flex items-center gap-2"
                >
                  {isCreating ? (
                    <>
                      <div className="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Plus className="h-4 w-4" />
                      Create Agent
                    </>
                  )}
                </Button>
                
                <Button 
                  variant="outline" 
                  onClick={resetForm}
                  className="flex items-center gap-2"
                >
                  Reset Form
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tools" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Available Tools</CardTitle>
              <CardDescription>
                Select the tools that your agent should be able to use
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {availableTools.map(tool => (
                  <div key={tool.name} className="border rounded-lg p-4 space-y-2">
                    <div className="flex items-start gap-2">
                      <Checkbox
                        id={`tool-${tool.name}`}
                        checked={selectedTools.includes(tool.name)}
                        onCheckedChange={() => handleToolToggle(tool.name)}
                      />
                      <div className="flex-1">
                        <label 
                          htmlFor={`tool-${tool.name}`}
                          className="text-sm font-medium cursor-pointer"
                        >
                          {tool.name}
                        </label>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                          {tool.description}
                        </p>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {tool.categories.map(category => (
                            <Badge key={category} variant="secondary" className="text-xs">
                              {category}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              {availableTools.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No tools available. Please check your backend configuration.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Generated Agent Code</span>
                <div className="flex gap-2">
                  {generatedCode && (
                    <>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleValidateCode}
                        disabled={isValidating}
                        className="flex items-center gap-2"
                      >
                        {isValidating ? (
                          <>
                            <div className="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full" />
                            Validating...
                          </>
                        ) : (
                          <>
                            <CheckCircle className="h-4 w-4" />
                            Validate
                          </>
                        )}
                      </Button>
                      
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={downloadCode}
                        className="flex items-center gap-2"
                      >
                        <Download className="h-4 w-4" />
                        Download
                      </Button>
                    </>
                  )}
                </div>
              </CardTitle>
              <CardDescription>
                Review and validate the generated agent code
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {generatedCode ? (
                <>
                  <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
                      <code>{generatedCode}</code>
                    </pre>
                  </div>
                  
                  {validationResult && (
                    <div className="space-y-3">
                      <Separator />
                      <div>
                        <h4 className="font-medium mb-2 flex items-center gap-2">
                          {validationResult.is_valid ? (
                            <>
                              <CheckCircle className="h-5 w-5 text-green-600" />
                              Validation Results
                            </>
                          ) : (
                            <>
                              <AlertCircle className="h-5 w-5 text-red-600" />
                              Validation Errors
                            </>
                          )}
                        </h4>
                        
                        {validationResult.errors.length > 0 && (
                          <div className="space-y-2">
                            <h5 className="font-medium text-red-600">Errors:</h5>
                            <ul className="list-disc list-inside space-y-1">
                              {validationResult.errors.map((error, index) => (
                                <li key={index} className="text-sm text-red-600">
                                  {error}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {validationResult.warnings.length > 0 && (
                          <div className="space-y-2">
                            <h5 className="font-medium text-yellow-600">Warnings:</h5>
                            <ul className="list-disc list-inside space-y-1">
                              {validationResult.warnings.map((warning, index) => (
                                <li key={index} className="text-sm text-yellow-600">
                                  {warning}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {validationResult.suggestions.length > 0 && (
                          <div className="space-y-2">
                            <h5 className="font-medium text-blue-600">Suggestions:</h5>
                            <ul className="list-disc list-inside space-y-1">
                              {validationResult.suggestions.map((suggestion, index) => (
                                <li key={index} className="text-sm text-blue-600">
                                  {suggestion}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Code className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <p>No agent code generated yet.</p>
                  <p className="text-sm">Go to the Design tab to create an agent.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="saved" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Saved Agents</CardTitle>
              <CardDescription>
                Manage your previously created custom agents
              </CardDescription>
            </CardHeader>
            <CardContent>
              {savedAgents.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {savedAgents.map(agent => (
                    <div key={agent.id} className="border rounded-lg p-4 space-y-3">
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="font-medium">{agent.name}</h3>
                          <Badge variant="secondary" className="mt-1">
                            {agent.category}
                          </Badge>
                        </div>
                        <div className="flex gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleLoadAgent(agent)}
                            className="h-8 w-8 p-0"
                          >
                            <Upload className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDeleteAgent(agent.id)}
                            className="h-8 w-8 p-0 text-red-600 hover:text-red-700"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      
                      <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                        {agent.description}
                      </p>
                      
                      <div className="text-xs text-gray-500">
                        Created: {new Date(agent.created_at).toLocaleDateString()}
                      </div>
                      
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleLoadAgent(agent)}
                          className="flex items-center gap-1"
                        >
                          <Code className="h-3 w-3" />
                          View Code
                        </Button>
                        
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleActivateAgent(agent.id)}
                          className="flex items-center gap-1"
                        >
                          <Play className="h-3 w-3" />
                          Activate
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Save className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <p>No saved agents yet.</p>
                  <p className="text-sm">Create your first agent in the Design tab.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AgentDesigner;