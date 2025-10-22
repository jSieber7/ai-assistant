import React from 'react';

const TestComponent: React.FC = () => {
  return (
    <div className="p-8 bg-blue-100 rounded-lg">
      <h1 className="text-2xl font-bold text-blue-800">Test Component</h1>
      <p className="mt-2 text-blue-600">If you can see this, React and Tailwind CSS are working!</p>
    </div>
  );
};

export default TestComponent;