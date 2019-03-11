define(["knockout", 'components/apply-now/academic/academic-viewmodel'], function(ko, academicVm) {
  function viewModel() {
    var self = this;
    self.testData = ko.observable("test data");
    self.isLoadingData = ko.observable(false);
    self.isVisible = ko.computed(function(){
      return academicVm().selectedAcademicLevel() != 'undergraduate' && !self.isLoadingData();
    });

    self.loadData = function(){
      self.isLoadingData(true);
      // mock ajax call for data
      var deferred = $.Deferred();
      setTimeout(function(){
        deferred.resolve();
        self.isLoadingData(false);
      }, 1200);

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
