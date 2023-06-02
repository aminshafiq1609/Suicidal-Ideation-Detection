document.addEventListener('input', function(event) {
    const element = event.target;
  
    if (element && element.matches('textarea[name="text"]')) {
      const text = element.value;
  
      if (isSuicidalContent(text)) {
        chrome.runtime.sendMessage({ message: 'showHotlines' });
      } else {
        chrome.runtime.sendMessage({ message: 'revertDisplay' });
      }
    }
  });
  