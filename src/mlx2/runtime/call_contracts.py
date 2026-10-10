"""Host-only keyword compatibility for a stable provider callable."""

import inspect


class KeywordSupportCache:
    """Bounded one-callable cache; method/provider replacement invalidates it.

    A true explicit declaration avoids introspection. Undeclared or false
    declarations preserve legacy signature discovery, including **kwargs.
    """

    def __init__(self, keyword):
        self.keyword = keyword
        self._function = self._receiver = self._signature = self._wrapped = None
        self._known = False
        self._supported = False
        self.inspections = 0

    def accepts(self, method, *, declared=False):
        if declared:
            return True
        function = getattr(method, "__func__", method)
        receiver = getattr(method, "__self__", None)
        signature = getattr(method, "__signature__", None)
        wrapped = getattr(method, "__wrapped__", None)
        if (
            self._known
            and function is self._function
            and receiver is self._receiver
            and signature is self._signature
            and wrapped is self._wrapped
        ):
            return self._supported
        self.inspections += 1
        try:
            parameters = inspect.signature(method).parameters
            named = parameters.get(self.keyword)
            supported = (
                named is not None
                and named.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ) or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
            )
        except (TypeError, ValueError):
            supported = False
        self._function, self._receiver = function, receiver
        self._signature, self._wrapped = signature, wrapped
        self._supported, self._known = supported, True
        return supported
