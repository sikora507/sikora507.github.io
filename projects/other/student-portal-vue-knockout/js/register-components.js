define(["knockout"], function(ko) {
  console.log('registering components');
  ko.components.register("apply-now", {
    template: { require: "text!/knockout-components/apply-now/apply-now-component.html" },
    viewModel: { require: "/knockout-components/apply-now/apply-now-viewmodel.js" }
  });
  ko.components.register("apply-now-basic-info", {
    template: { require: "text!/knockout-components/apply-now/basic-info/basic-info-component.html" },
    viewModel: { require: "/knockout-components/apply-now/basic-info/basic-info-viewmodel.js" }
  });
  ko.components.register("apply-now-academic", {
    template: { require: "text!/knockout-components/apply-now/academic/academic-component.html" },
    viewModel: { require: "/knockout-components/apply-now/academic/academic-viewmodel.js" }
  });
  ko.components.register("apply-now-highschool-info", {
    template: { require: "text!/knockout-components/apply-now/highschool-info/highschool-info-component.html" },
    viewModel: { require: "/knockout-components/apply-now/highschool-info/highschool-info-viewmodel.js" }
  });
  ko.components.register("apply-now-navigation-bar", {
    template: { require: "text!/knockout-components/apply-now/navigation-bar/navigation-bar-component.html" },
    viewModel: { require: "/knockout-components/apply-now/navigation-bar/navigation-bar-viewmodel.js" }
  });
  ko.components.register("loader", {
    template: { require: "text!/knockout-components/loader/loader-component.html" },
  });
});
