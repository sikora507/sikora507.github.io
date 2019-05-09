define(['knockout'], function(ko){
    var messageBus = new ko.subscribable();
    
    messageBus.onEvent = function(eventName, callback){
        messageBus.subscribe(callback, null, eventName)
    };
    messageBus.sendEvent = function(eventName, data){
        messageBus.notifySubscribers(data, eventName);
    };
    
    return messageBus;
});