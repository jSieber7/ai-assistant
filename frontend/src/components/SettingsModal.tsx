import React from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import * as Slider from '@radix-ui/react-slider';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Settings } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  temperature: number;
  maxTokens: number;
  onTemperatureChange: (value: number[]) => void;
  onMaxTokensChange: (value: number[]) => void;
  onReset: () => void;
  onSave: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  temperature,
  maxTokens,
  onTemperatureChange,
  onMaxTokensChange,
  onReset,
  onSave,
}) => {
  const handleReset = () => {
    onTemperatureChange([0.7]);
    onMaxTokensChange([0]);
    onReset();
  };

  return (
    <Dialog.Root open={isOpen} onOpenChange={onClose}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
        <Dialog.Content className="fixed left-[50%] top-[50%] translate-x-[-50%] translate-y-[-50%] w-full max-w-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%]">
          <Card className="w-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Chat Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">Temperature</label>
                  <span className="text-sm text-gray-500">{temperature.toFixed(1)}</span>
                </div>
                <Slider.Root
                  className="relative flex items-center select-none touch-none w-full h-5"
                  value={[temperature]}
                  onValueChange={onTemperatureChange}
                  max={2}
                  min={0}
                  step={0.1}
                >
                  <Slider.Track className="bg-gray-200 relative grow rounded-full h-[3px]">
                    <Slider.Range className="absolute bg-blue-500 rounded-full h-full" />
                  </Slider.Track>
                  <Slider.Thumb
                    className="block w-4 h-4 bg-white shadow-lg rounded-full hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    aria-label="Temperature"
                  />
                </Slider.Root>
                <p className="text-xs text-gray-500">
                  Controls randomness: Lower values make responses more focused and deterministic, higher values make them more creative.
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">Max Tokens</label>
                  <span className="text-sm text-gray-500">{maxTokens === 0 ? 'Unlimited' : maxTokens}</span>
                </div>
                <Slider.Root
                  className="relative flex items-center select-none touch-none w-full h-5"
                  value={[maxTokens]}
                  onValueChange={onMaxTokensChange}
                  max={4000}
                  min={0}
                  step={100}
                >
                  <Slider.Track className="bg-gray-200 relative grow rounded-full h-[3px]">
                    <Slider.Range className="absolute bg-blue-500 rounded-full h-full" />
                  </Slider.Track>
                  <Slider.Thumb
                    className="block w-4 h-4 bg-white shadow-lg rounded-full hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    aria-label="Max Tokens"
                  />
                </Slider.Root>
                <p className="text-xs text-gray-500">
                  Maximum number of tokens to generate. Set to 0 for unlimited.
                </p>
              </div>

              <div className="flex justify-between pt-4">
                <Button variant="outline" onClick={handleReset}>
                  Reset
                </Button>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={onClose}>
                    Cancel
                  </Button>
                  <Button onClick={onSave}>
                    Continue
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};

export default SettingsModal;