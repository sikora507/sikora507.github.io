define([
    "knockout",
    "message-bus"
  ], function(ko, messageBus) {
    return function appViewModel() {
      var self = this;
      self.contentComponent = ko.observable("home-page");
      messageBus.onEvent('menu-item-changed', function(data){
        self.contentComponent(data);
      });
      self.setContent = function(componentName) {
        self.contentComponent(componentName);
      };
    };
  });
  