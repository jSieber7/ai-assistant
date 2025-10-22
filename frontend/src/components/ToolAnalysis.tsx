import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { 
  Wrench, 
  RefreshCw, 
  Power, 
  PowerOff, 
  Info, 
  BarChart3, 
  Settings, 
  Play,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react';
import { useTools } from '../hooks/useTools';

const ToolAnalysis: React.FC = () => {
  const {
    tools,
    selectedTool,
    registryStats,
    categories,
    toolsByCategory,
    isLoading,
    error,
    loadTools,
    loadToolInfo,
    loadRegistryStats,
    loadToolsByCategory,
    enableTool,
    disableTool,
    executeTool,
    clearSelectedTool,
    loadAll,
  } = useTools();

  const [executionParams, setExecutionParams] = useState<Record<string, any>>({});
  const [executionResult, setExecutionResult] = useState<any>(null);
  const [isExecuting, setIsExecuting] = useState(false);

  const handleToolToggle = async (toolName: string, enabled: boolean) => {
    if (enabled) {
      await enableTool(toolName);
    } else {
      await disableTool(toolName);
    }
  };

  const handleToolSelect = async (toolName: string) => {
    await loadToolInfo(toolName);
    setExecutionParams({});
    setExecutionResult(null);
  };

  const handleExecuteTool = async () => {
    if (!selectedTool) return;
    
    setIsExecuting(true);
    try {
      const result = await executeTool(selectedTool.name, executionParams);
      setExecutionResult(result);
    } catch (error) {
      console.error('Tool execution failed:', error);
    } finally {
      setIsExecuting(false);
    }
  };

  const handleParamChange = (paramName: string, value: any) => {
    setExecutionParams(prev => ({
      ...prev,
      [paramName]: value,
    }));
  };

  const renderParameterInput = (paramName: string, paramConfig: any) => {
    const { type, description, default: defaultValue } = paramConfig;
    
    return (
      <div key={paramName} className="space-y-2">
        <label className="text-sm font-medium">{paramName}</label>
        {description && <p className="text-xs text-gray-500">{description}</p>}
        
        {type === 'string' && (
          <input
            type="text"
            value={executionParams[paramName] || defaultValue || ''}
            onChange={(e) => handleParamChange(paramName, e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder={`Enter ${paramName}`}
          />
        )}
        
        {type === 'number' && (
          <input
            type="number"
            value={executionParams[paramName] || defaultValue || ''}
            onChange={(e) => handleParamChange(paramName, Number(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder={`Enter ${paramName}`}
          />
        )}
        
        {type === 'boolean' && (
          <Switch
            checked={executionParams[paramName] ?? defaultValue ?? false}
            onCheckedChange={(checked) => handleParamChange(paramName, checked)}
          />
        )}
      </div>
    );
  };

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <CardTitle>Error</CardTitle>
            <CardDescription>{error}</CardDescription>
          </CardHeader>
          <CardContent className="text-center">
            <Button onClick={loadAll} className="w-full">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-2">
              <Wrench className="h-8 w-8" />
              Tool Analysis
            </h1>
            <p className="text-gray-500 dark:text-gray-400 mt-2">
              Analyze and manage available tools in the system
            </p>
          </div>
          <Button onClick={loadAll} disabled={isLoading} variant="outline">
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        {/* Registry Statistics */}
        {registryStats && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Registry Statistics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div className="text-2xl font-bold">{registryStats.total_tools}</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">Total Tools</div>
                </div>
                <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {registryStats.enabled_tools}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">Enabled</div>
                </div>
                <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                    {registryStats.disabled_tools}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">Disabled</div>
                </div>
              </div>
              
              {Object.keys(registryStats.categories).length > 0 && (
                <div className="mt-4">
                  <h4 className="font-medium mb-2">Categories</h4>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(registryStats.categories).map(([category, count]) => (
                      <Badge key={category} variant="secondary">
                        {category}: {count}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        <Tabs defaultValue="tools" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="tools">All Tools</TabsTrigger>
            <TabsTrigger value="categories">Categories</TabsTrigger>
            <TabsTrigger value="details">Tool Details</TabsTrigger>
          </TabsList>

          {/* All Tools Tab */}
          <TabsContent value="tools" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {tools.map((tool) => (
                <Card key={tool.name} className="cursor-pointer hover:shadow-md transition-shadow">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{tool.name}</CardTitle>
                      <div className="flex items-center gap-2">
                        {tool.enabled ? (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        ) : (
                          <PowerOff className="h-5 w-5 text-red-500" />
                        )}
                        <Switch
                          checked={tool.enabled}
                          onCheckedChange={(checked) => handleToolToggle(tool.name, checked)}
                        />
                      </div>
                    </div>
                    <CardDescription>{tool.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-between">
                      <div className="flex flex-wrap gap-1">
                        {tool.categories.map((category) => (
                          <Badge key={category} variant="outline" className="text-xs">
                            {category}
                          </Badge>
                        ))}
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleToolSelect(tool.name)}
                      >
                        <Info className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Categories Tab */}
          <TabsContent value="categories" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {categories.map((category) => (
                <Card key={category}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Settings className="h-5 w-5" />
                      {category}
                    </CardTitle>
                    <CardDescription>
                      {registryStats?.categories[category] || 0} tools in this category
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Button
                      variant="outline"
                      className="w-full"
                      onClick={() => loadToolsByCategory(category)}
                      disabled={isLoading}
                    >
                      View Tools
                    </Button>
                    
                    {toolsByCategory[category] && (
                      <div className="mt-4 space-y-2">
                        {toolsByCategory[category].map((tool) => (
                          <div
                            key={tool.name}
                            className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded"
                          >
                            <div className="flex items-center gap-2">
                              {tool.enabled ? (
                                <CheckCircle className="h-4 w-4 text-green-500" />
                              ) : (
                                <PowerOff className="h-4 w-4 text-red-500" />
                              )}
                              <span className="text-sm">{tool.name}</span>
                            </div>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleToolSelect(tool.name)}
                            >
                              <Info className="h-4 w-4" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Tool Details Tab */}
          <TabsContent value="details" className="space-y-4">
            {selectedTool ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Tool Information */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Info className="h-5 w-5" />
                      {selectedTool.name}
                    </CardTitle>
                    <CardDescription>{selectedTool.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium">Version:</span>
                        <p>{selectedTool.version}</p>
                      </div>
                      <div>
                        <span className="font-medium">Author:</span>
                        <p>{selectedTool.author}</p>
                      </div>
                      <div>
                        <span className="font-medium">Timeout:</span>
                        <p>{selectedTool.timeout}s</p>
                      </div>
                      <div>
                        <span className="font-medium">Status:</span>
                        <p>
                          {selectedTool.enabled ? (
                            <Badge variant="default" className="bg-green-500">
                              Enabled
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="text-red-500">
                              Disabled
                            </Badge>
                          )}
                        </p>
                      </div>
                    </div>

                    <Separator />

                    <div>
                      <h4 className="font-medium mb-2">Categories</h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedTool.categories.map((category) => (
                          <Badge key={category} variant="secondary">
                            {category}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium mb-2">Keywords</h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedTool.keywords.map((keyword) => (
                          <Badge key={keyword} variant="outline">
                            {keyword}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Tool Execution */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Play className="h-5 w-5" />
                      Execute Tool
                    </CardTitle>
                    <CardDescription>
                      Test the tool with custom parameters
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {Object.entries(selectedTool.parameters).map(([paramName, paramConfig]) =>
                      renderParameterInput(paramName, paramConfig)
                    )}

                    <Button
                      onClick={handleExecuteTool}
                      disabled={!selectedTool.enabled || isExecuting}
                      className="w-full"
                    >
                      {isExecuting ? (
                        <>
                          <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                          Executing...
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4 mr-2" />
                          Execute Tool
                        </>
                      )}
                    </Button>

                    {executionResult && (
                      <div className="mt-4">
                        <h4 className="font-medium mb-2 flex items-center gap-2">
                          <Clock className="h-4 w-4" />
                          Execution Result
                        </h4>
                        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
                          <div className="text-sm mb-2">
                            <span className="font-medium">Success:</span>{' '}
                            {executionResult.success ? (
                              <Badge variant="default" className="bg-green-500 ml-1">
                                Yes
                              </Badge>
                            ) : (
                              <Badge variant="destructive" className="ml-1">
                                No
                              </Badge>
                            )}
                          </div>
                          <div className="text-sm mb-2">
                            <span className="font-medium">Execution Time:</span>{' '}
                            {executionResult.execution_time}s
                          </div>
                          {executionResult.error && (
                            <div className="text-sm mb-2">
                              <span className="font-medium">Error:</span>{' '}
                              <span className="text-red-500">{executionResult.error}</span>
                            </div>
                          )}
                          <div className="text-sm">
                            <span className="font-medium">Data:</span>
                            <pre className="mt-1 p-2 bg-gray-100 dark:bg-gray-900 rounded text-xs overflow-auto">
                              {JSON.stringify(executionResult.data, null, 2)}
                            </pre>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <Wrench className="h-12 w-12 text-gray-400 mb-4" />
                  <h3 className="text-lg font-medium mb-2">No Tool Selected</h3>
                  <p className="text-gray-500 text-center mb-4">
                    Select a tool from the All Tools or Categories tab to view detailed information
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default ToolAnalysis;