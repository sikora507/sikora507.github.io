define(["knockout"], function(ko) {
  ko.components.register("main-menu", {
    template: { require: "text!components/main-menu/main-menu-component.html" },
    viewModel: { require: "components/main-menu/main-menu-viewmodel" }
  });
  ko.components.register("home-page", {
    template: { require: "text!components/home-page/home-page-component.html" }
  });
  ko.components.register("apply-now", {
    template: { require: "text!components/apply-now/apply-now-component.html" },
    viewModel: { require: "components/apply-now/apply-now-viewmodel" }
  });
  ko.components.register("apply-now-basic-info", {
    template: { require: "text!components/apply-now/basic-info/basic-info-component.html" },
    viewModel: { require: "components/apply-now/basic-info/basic-info-viewmodel" }
  });
  ko.components.register("apply-now-academic", {
    template: { require: "text!components/apply-now/academic/academic-component.html" },
    viewModel: { require: "components/apply-now/academic/academic-viewmodel" }
  });
  ko.components.register("apply-now-highschool-info", {
    template: { require: "text!components/apply-now/highschool-info/highschool-info-component.html" },
    viewModel: { require: "components/apply-now/highschool-info/highschool-info-viewmodel" }
  });
  ko.components.register("apply-now-navigation-bar", {
    template: { require: "text!components/apply-now/navigation-bar/navigation-bar-component.html" },
    viewModel: { require: "components/apply-now/navigation-bar/navigation-bar-viewmodel" }
  });
  ko.components.register("loader", {
    template: { require: "text!components/loader/loader-component.html" }
  });
});
