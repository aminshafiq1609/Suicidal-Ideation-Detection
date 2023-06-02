// Listen for messages from the extension button
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    // Check if the message is to show the popup
    if (request.message === 'showPopup') {
      // Display the popup
      chrome.browserAction.setPopup({ popup: 'popup.html' });
      chrome.browserAction.openPopup();
    }
  });
  