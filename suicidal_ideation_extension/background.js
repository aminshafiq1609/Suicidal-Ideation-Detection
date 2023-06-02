chrome.runtime.onInstalled.addListener(function() {
    console.log('Extension installed');
  });
  
  chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.message === 'showHotlines') {
      console.log('Display hotlines');
      // You can update the DOM to display the hotlines or perform any other desired action
    } else if (request.message === 'revertDisplay') {
      console.log('Revert to default display');
      // You can revert the DOM to the default display or perform any other desired action
    }
  });
  
  