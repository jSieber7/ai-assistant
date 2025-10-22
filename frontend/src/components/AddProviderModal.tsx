import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Plus } from 'lucide-react';
import { showToast } from '@/lib/toast';

// Form validation schema
const providerFormSchema = z.object({
  name: z.string().min(1, 'Provider name is required'),
  type: z.string().min(1, 'Provider type is required'),
  api_key: z.string().optional(),
  api_base: z.string().optional(),
  is_default: z.boolean().default(false),
});

type ProviderFormValues = z.infer<typeof providerFormSchema>;

interface AddProviderRequest {
  name: string;
  type: string;
  api_key?: string;
  api_base?: string;
  is_default?: boolean;
  [key: string]: any;
}

interface AddProviderModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAddProvider: (providerData: AddProviderRequest) => Promise<{ success: boolean; message: string }>;
  isLoading?: boolean;
}

const AddProviderModal: React.FC<AddProviderModalProps> = ({
  isOpen,
  onClose,
  onAddProvider,
  isLoading = false,
}) => {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const form = useForm<ProviderFormValues>({
    resolver: zodResolver(providerFormSchema),
    defaultValues: {
      name: '',
      type: '',
      api_key: '',
      api_base: '',
      is_default: false,
    },
  });

  const providerTypes = [
    { value: 'openai', label: 'OpenAI' },
    { value: 'anthropic', label: 'Anthropic' },
    { value: 'ollama', label: 'Ollama' },
    { value: 'llama.cpp', label: 'Llama.cpp' },
    { value: 'azure', label: 'Azure OpenAI' },
    { value: 'custom', label: 'Custom' },
  ];

  const onSubmit = async (data: ProviderFormValues) => {
    setIsSubmitting(true);
    
    try {
      const providerData: AddProviderRequest = {
        ...data,
      };

      const result = await onAddProvider(providerData);
      
      if (result.success) {
        showToast.success(result.message);
        form.reset();
        onClose();
      } else {
        showToast.error(result.message);
      }
    } catch (error) {
      showToast.error('Failed to add provider');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleProviderTypeChange = (value: string) => {
    form.setValue('type', value);
    
    // Set default values based on provider type
    switch (value) {
      case 'openai':
        form.setValue('api_base', 'https://api.openai.com/v1');
        break;
      case 'anthropic':
        form.setValue('api_base', 'https://api.anthropic.com');
        break;
      case 'ollama':
        form.setValue('api_base', 'http://localhost:11434');
        form.setValue('api_key', ''); // Clear API key for local providers
        break;
      case 'llama.cpp':
        form.setValue('api_base', 'http://localhost:8080');
        form.setValue('api_key', ''); // Clear API key for local providers
        break;
      case 'azure':
        form.setValue('api_base', '');
        break;
      default:
        form.setValue('api_base', '');
    }
  };

  // Check if API key is required for the selected provider type
  const isApiKeyRequired = (providerType: string) => {
    const localProviders = ['ollama', 'llama.cpp'];
    return !localProviders.includes(providerType);
  };

  const selectedProviderType = form.watch('type');

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Add Model Provider</DialogTitle>
          <DialogDescription>
            Configure a new model provider to use with the AI assistant.
          </DialogDescription>
        </DialogHeader>
        
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Provider Name</Label>
            <Input
              id="name"
              placeholder="e.g., OpenAI GPT-4"
              {...form.register('name')}
            />
            {form.formState.errors.name && (
              <p className="text-sm text-red-500">{form.formState.errors.name.message}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="type">Provider Type</Label>
            <Select onValueChange={handleProviderTypeChange} defaultValue={form.getValues('type')}>
              <SelectTrigger>
                <SelectValue placeholder="Select provider type" />
              </SelectTrigger>
              <SelectContent>
                {providerTypes.map((type) => (
                  <SelectItem key={type.value} value={type.value}>
                    {type.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {form.formState.errors.type && (
              <p className="text-sm text-red-500">{form.formState.errors.type.message}</p>
            )}
          </div>

          {isApiKeyRequired(selectedProviderType) && (
            <div className="space-y-2">
              <Label htmlFor="api_key">API Key</Label>
              <Input
                id="api_key"
                type="password"
                placeholder="Enter API key"
                {...form.register('api_key')}
              />
              {form.formState.errors.api_key && (
                <p className="text-sm text-red-500">{form.formState.errors.api_key.message}</p>
              )}
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="api_base">API Base URL</Label>
            <Input
              id="api_base"
              placeholder="https://api.example.com/v1"
              {...form.register('api_base')}
            />
            {form.formState.errors.api_base && (
              <p className="text-sm text-red-500">{form.formState.errors.api_base.message}</p>
            )}
          </div>


          <div className="flex items-center space-x-2">
            <Switch
              id="is_default"
              checked={form.watch('is_default')}
              onCheckedChange={(checked) => form.setValue('is_default', checked)}
            />
            <Label htmlFor="is_default">Set as default provider</Label>
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting || isLoading}>
              {isSubmitting ? 'Adding...' : 'Add Provider'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default AddProviderModal;