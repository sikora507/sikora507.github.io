requirejs.config({
  baseUrl: "/js",
  paths: {
    components: "/knockout-components",
    knockout: "/js/external/knockout-3.5.0",
    text: "/js/external/text",
    domReady: "/js/external/domReady",
    jquery: "/js/external/jquery-3.3.1.min"
  }
});

require([
  //"knockout",
  //"app-viewmodel",
  "domReady!",
  "register-components",
  "message-bus"
], function(/*ko, appViewModel*/) {
  //ko.applyBindings(new appViewModel());
});
