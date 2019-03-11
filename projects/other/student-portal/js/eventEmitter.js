define([], function(){
    return {
        emit: function(element, name, data){
            let event = new CustomEvent(name, {bubbles: true, detail: data});
            element.dispatchEvent(event);
        }
    }
})