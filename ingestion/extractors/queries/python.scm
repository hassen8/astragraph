;; @query: function
(function_definition) @fn.def

;; @query: class
(class_definition) @cls.def

;; @query: import
(import_statement) @import
(import_from_statement) @import

;; @query: call
(call
  function: (identifier) @call.name) @call.site

(call
  function: (attribute
    object: (_) @call.object
    attribute: (identifier) @call.attr)) @call.site
