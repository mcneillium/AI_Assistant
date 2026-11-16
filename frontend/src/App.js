import React, { useState } from 'react';

function App() {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');

  const handleGenerateText = async () => {
    const response = await fetch('http://127.0.0.1:5000/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input_text: input }),
    });
    const data = await response.json();
    setOutput(response.ok ? data.generated_text : `Error: ${data.error}`);
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>AI Assistant</h1>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter some text"
        style={{ padding: '10px', width: '300px', marginRight: '10px' }}
      />
      <button onClick={handleGenerateText} style={{ padding: '10px' }}>
        Generate
      </button>
      <div style={{ marginTop: '20px' }}>
        <h2>Generated Text:</h2>
        <p>{output}</p>
      </div>
    </div>
  );
}

export default App;
