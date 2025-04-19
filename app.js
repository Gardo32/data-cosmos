// ...existing code...

document.getElementById('submit-btn').addEventListener('click', async () => {
    const promptInput = document.getElementById('prompt-input').value;
    if (!promptInput.trim()) return;

    const responseContainer = document.getElementById('response-container');
    const reportContainer = document.getElementById('report-container');
    const loadingElement = document.getElementById('loading');
    
    responseContainer.innerHTML = '';
    reportContainer.innerHTML = '';
    reportContainer.classList.add('hidden');
    loadingElement.classList.remove('hidden');
    
    try {
        const response = await fetch('/api/prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: promptInput }),
        });
        
        const data = await response.json();
        loadingElement.classList.add('hidden');
        
        if (data.error) {
            responseContainer.innerHTML = `<div class="error">${data.error}</div>`;
            return;
        }
        
        // Process the response
        responseContainer.innerHTML = processResponseText(data.response);
        
        // Extract and display report if present
        const reportMatch = data.response.match(/<report>([\s\S]*?)<\/report>/);
        if (reportMatch && reportMatch[1]) {
            const reportContent = reportMatch[1].trim();
            reportContainer.innerHTML = marked.parse(reportContent);
            reportContainer.classList.remove('hidden');
        }
    } catch (error) {
        loadingElement.classList.add('hidden');
        responseContainer.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
});

function processResponseText(text) {
    // Remove report sections from the main display
    const cleanedText = text.replace(/<report>[\s\S]*?<\/report>/g, '');
    
    // Format the rest of the text
    return cleanedText
        .replace(/\n/g, '<br>')
        .replace(/```(\w*)([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
}

// ...existing code...