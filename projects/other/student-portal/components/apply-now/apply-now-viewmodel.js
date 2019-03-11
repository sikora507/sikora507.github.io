define([
  "knockout",
  "components/apply-now/basic-info/basic-info-viewmodel",
  "components/apply-now/academic/academic-viewmodel",
  "components/apply-now/highschool-info/highschool-info-viewmodel"
], function (ko, basicInfoVm, academicVm, highschoolInfoVm) {
  function viewModel() {
    var self = this;
    self.components = ko.observableArray([
      { name: 'Basic info', componentName: 'apply-now-basic-info', vm: basicInfoVm() },
      { name: 'Academic', componentName: 'apply-now-academic', vm: academicVm() },
      { name: 'High School Info', componentName: 'apply-now-highschool-info', vm: highschoolInfoVm() }
    ]);
    self.visibleComponents = ko.computed(function(){
      let filteredComponents = self.components().filter(function(c){return !c.vm.isVisible || c.vm.isVisible && c.vm.isVisible()});
      return filteredComponents;
    })
    self.selectedSection = ko.observable(self.components()[0]);
    self.selectApplyNowSection = function (component) {
      self.selectedSection(component);
    }
    self.selectedComponentName = ko.computed(function () {
      return self.selectedSection().componentName;
    });
    self.goToNextApplyNowStep = function () {
      if (self.canGoToNextStep()) {
        let index = getSelectedComponentIndex();
        self.selectedSection(self.visibleComponents()[index + 1]);
      }
    }
    self.goToPrevApplyNowStep = function () {
      if (self.canGoToPrevStep()) {
        let index = getSelectedComponentIndex();
        self.selectedSection(self.visibleComponents()[index - 1]);
      }
    }

    self.canGoToNextStep = ko.computed(function () {
      let index = getSelectedComponentIndex();
      if (index < self.visibleComponents().length - 1) {
        return true;
      }
      return false;
    });

    self.canGoToPrevStep = ko.computed(function () {
      let index = getSelectedComponentIndex();
      if (index > 0) {
        return true;
      }
      return false;
    });

    function getSelectedComponentIndex() {
      let index = self.visibleComponents().indexOf(self.selectedSection());
      return index;
    }
  }
  instance = new viewModel();
  return function () { return instance; };
});
