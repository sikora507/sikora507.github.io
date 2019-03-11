define(["knockout", 'eventEmitter'], function (ko, emitter) {
  ko.bindingHandlers.clickEmit = {
    init: function (element, valueAccessor) {
      var accessor = valueAccessor();

      element.addEventListener("click", function () {
        let data = null;
        let name = null;
        if (typeof (accessor) === 'object') {
          data = accessor.data;
          name = accessor.name;
        } else {
          name = accessor;
        }
        emitter.emit(element, name, data);
      });
    }
  };
});
