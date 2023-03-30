// Website with idea: https://g3rv4.com/2017/08/shiny-detect-mobile-browsers
var isMobileBinding = new Shiny.InputBinding();
$.extend(isMobileBinding, {
  find: function(scope) {
    return $(scope).find(".mobile-element");
    callback();
  },
  getValue: function(el) {
    return /((iPhone)|(iPod)|(iPad)|(Android)|(BlackBerry))/.test(navigator.userAgent)
  },
  setValue: function(el, value) {
  },
  subscribe: function(el, callback) {
  },
  unsubscribe: function(el) {
  }
});

Shiny.inputBindings.register(isMobileBinding);

function createCORSRequest(method, url){
  var xhr = new XMLHttpRequest();
  if ("withCredentials" in xhr){
    // XHR has 'withCredentials' property only if it supports CORS
    xhr.open(method, url, true);
    xhr.timeout = 2000;
  } else if (typeof XDomainRequest != "undefined"){ // if IE use XDR
    xhr = new XDomainRequest();
    xhr.open(method, url);
    xhr.timeout = 2000;
  } else {
    return "fail";
  }
  
  return xhr;
}


//User time
$(document).on('shiny:connected', function(event) {
  var now = new Date().toLocaleString('en-us', {timeZoneName:'short'});
  Shiny.setInputValue("clientTime", now);
  
  var request = createCORSRequest( "get", "https://ipapi.co/json/" );
  if ( request != "fail" ){
    
    // Define a callback function
    request.onload = function(){
  	 if(request.status == 200){
        Shiny.setInputValue("ipLoc", request.responseText);
      } else {
        Shiny.setInputValue("ipLoc", "fail");
      }
    };
    
    // On timeout
    request.ontimeout = function (e) {
      Shiny.setInputValue("ipLoc", "fail");
    };
    
    // Send request
    request.send();
  } else {
    Shiny.setInputValue("ipLoc", "fail");
  }
  
});

//$(document).ready(function(){
//  var header = $('.navbar> .container-fluid > .navbar-collapse');
//  header.append('<img src="headerLogo.png" align="right" height="40px">');

//});


  

