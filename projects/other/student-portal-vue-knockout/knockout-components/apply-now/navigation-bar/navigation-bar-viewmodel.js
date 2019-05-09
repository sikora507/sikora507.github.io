define(["knockout", 'message-bus'], function (ko, messageBus) {
    return function viewModel(params) {
        var self = this;
        self.prevActive = ko.computed(function () {
            return params.prevActive();
        });
        self.nextActive = ko.computed(function () {
            return params.nextActive();
        });
        self.goToNextApplyNowStep = function(){
            messageBus.sendEvent('go-to-next-apply-now-step');
        };
        self.goToPrevApplyNowStep = function(){
            messageBus.sendEvent('go-to-prev-apply-now-step');
        };
    }
});
