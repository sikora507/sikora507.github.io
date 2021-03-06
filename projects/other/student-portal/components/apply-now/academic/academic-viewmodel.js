define(["knockout", "jquery"], function(ko, $) {
  function viewModel() {
    var self = this;
    //self.isActive = ko.observable(true);
    self.testData = ko.observable("test data");
    self.academicLevels = ko.observableArray([
      {name: 'Select One', value: null},
      {name: 'Graduate', value: 'graduate'},
      {name: 'Undergraduate', value: 'undergraduate'},
    ]);
    self.selectedAcademicLevel = ko.observable(null);
    
    self.isVisible = ko.observable(false);
    self.loadData = function(){
      // mock ajax call for data
      self.isVisible(false);
      var deferred = $.Deferred();
      setTimeout(function(){
        self.isVisible(true);
        deferred.resolve();
      }, 1000);
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
