define(["knockout", "jquery"], function(ko, $) {
  function viewModel() {
    var self = this;

    self.isVisible = ko.observable(false);

    self.loadData = function(){
      // mock ajax call for data
      self.isVisible(false);
      var deferred = $.Deferred();
      setTimeout(function(){
        self.isVisible(true);
        deferred.resolve();
      }, 800);
      return deferred.promise();
    }
    function constructor(){
      self.loadData();
    }
    constructor();
  }
  var instance = new viewModel();
  return function(){return instance;};
});
