define([
    "knockout"
  ], function(ko) {
    return function appViewModel() {
      this.contentComponent = ko.observable("home-page");
      this.setContent = function(componentName) {
        this.contentComponent(componentName);
      };
    };
  });
  