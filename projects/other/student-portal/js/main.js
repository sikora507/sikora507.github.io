requirejs.config({
  baseUrl: "/projects/other/student-portal/js",
  paths: {
    components: "/projects/other/student-portal/components",
    knockout: "/projects/other/student-portal/js/external/knockout-3.5.0",
    text: "/projects/other/student-portal/js/external/text",
    domReady: "/projects/other/student-portal/js/external/domReady",
    jquery: "/projects/other/student-portal/js/external/jquery-3.3.1.min"
  }
});

require([
  "knockout",
  "app-viewmodel",
  "domReady!",
  "register-components",
  "register-bindings"
], function(ko, appViewModel) {
  ko.applyBindings(new appViewModel());
});
