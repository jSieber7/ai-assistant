import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Mic, Square } from 'lucide-react';

interface InputAreaProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  onStopGeneration: () => void;
  disabled?: boolean;
}

const InputArea: React.FC<InputAreaProps> = ({
  onSendMessage,
  isLoading,
  onStopGeneration,
  disabled = false,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (inputValue.trim() && !isLoading && !disabled) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileUpload = () => {
    // TODO: Implement file upload functionality
    console.log('File upload not implemented yet');
  };

  const handleVoiceRecording = () => {
    // TODO: Implement voice recording functionality
    setIsRecording(!isRecording);
    console.log('Voice recording not implemented yet');
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [inputValue]);

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-end gap-3">
          {/* File Upload Button */}
          <button
            onClick={handleFileUpload}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            title="Upload file"
          >
            <Paperclip className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          </button>

          {/* Input Field */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={disabled ? "Please select a model or agent first" : "Type your message..."}
              disabled={disabled || isLoading}
              className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              rows={1}
              style={{ minHeight: '48px', maxHeight: '200px' }}
            />
          </div>

          {/* Voice Recording Button */}
          <button
            onClick={handleVoiceRecording}
            className={`p-2 rounded-lg transition-colors ${
              isRecording
                ? 'bg-red-100 dark:bg-red-900 text-red-600 dark:text-red-400'
                : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400'
            }`}
            title="Voice input"
          >
            <Mic className="h-5 w-5" />
          </button>

          {/* Send/Stop Button */}
          <button
            onClick={isLoading ? onStopGeneration : handleSend}
            disabled={!inputValue.trim() && !isLoading}
            className={`p-2 rounded-lg transition-colors ${
              isLoading
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : inputValue.trim() && !disabled
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
            }`}
            title={isLoading ? "Stop generation" : "Send message"}
          >
            {isLoading ? (
              <Square className="h-5 w-5" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </button>
        </div>

        {/* Input Status */}
        <div className="flex items-center justify-between mt-2 text-xs text-gray-500 dark:text-gray-400">
          <div>
            {isRecording && (
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
                Recording...
              </span>
            )}
          </div>
          <div>
            {disabled ? (
              <span>Select a model or agent to start chatting</span>
            ) : (
              <span>Press Enter to send, Shift+Enter for new line</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default InputArea;