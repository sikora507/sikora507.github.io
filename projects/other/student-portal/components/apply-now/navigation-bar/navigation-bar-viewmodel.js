define(["knockout"], function (ko) {
    return function viewModel(params) {
        var self = this;
        self.prevActive = ko.computed(function () {
            return params.prevActive();
        });
        self.nextActive = ko.computed(function () {
            return params.nextActive();
        })
    }
});
