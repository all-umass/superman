import shlex

__all__ = ['parse_query']

_QUERY_CACHE = {}  # (query, case) -> parsed fn
_CACHE_MISS = object()  # sentinel object, doesn't do anything


def parse_query(query, case_sensitive=False):
  cache_key = (query, case_sensitive)
  res = _QUERY_CACHE.get(cache_key, _CACHE_MISS)
  if res is not _CACHE_MISS:
    return res
  res = _parse_query(query, case_sensitive)
  # TODO: limit the size of the cache, LRU-style
  _QUERY_CACHE[cache_key] = res
  return res


def _parse_query(query, case_sensitive):
  tokens = shlex.split(query)
  if not tokens:
    return None

  # binding precedence: NOT > AND > OR > adjacency (implicit AND)
  op_precedence = [(u'NOT', NotExpr), (u'AND', AndExpr), (u'OR', OrExpr)]
  for lexeme, expr_type in op_precedence:
    while True:
      idx = _find_keyword(lexeme, tokens)
      if idx is None:
        break
      rhs = tokens[idx+1]
      if expr_type.arity == 2:
        lhs = tokens[idx-1]
        tokens = tokens[:idx-1] + [expr_type(lhs, rhs)] + tokens[idx+2:]
      else:  # unary expr
        tokens = tokens[:idx] + [expr_type(rhs)] + tokens[idx+2:]

  # AND together any remaining roots
  while len(tokens) > 1:
    rhs = tokens.pop()
    lhs = tokens.pop()
    tokens.append(AndExpr(lhs, rhs))

  # get a matcher function from the root
  root, = tokens
  return _matcher(root, case_sensitive)


def _find_keyword(lexeme, tokens):
  for i, tok in enumerate(tokens):
    if not isinstance(tok, QueryExpr) and tok.upper() == lexeme:
      return i
  return None


def _matcher(token, case_sensitive):
  if isinstance(token, QueryExpr):
    return token.matcher(case=case_sensitive)
  elif case_sensitive:
    return lambda text: token in text
  else:
    low = token.lower()
    return lambda text: low in text.lower()


class QueryExpr(object):
  def __init__(self, arg1, arg2=None):
    self.arg1 = arg1
    self.arg2 = arg2


class AndExpr(QueryExpr):
  arity = 2

  def matcher(self, case=False):
    lhs = _matcher(self.arg1, case)
    rhs = _matcher(self.arg2, case)
    return lambda text: lhs(text) and rhs(text)

  def __repr__(self):
    return u'(AND: %r %r)' % (self.arg1, self.arg2)


class OrExpr(QueryExpr):
  arity = 2

  def matcher(self, case=False):
    lhs = _matcher(self.arg1, case)
    rhs = _matcher(self.arg2, case)
    return lambda text: lhs(text) or rhs(text)

  def __repr__(self):
    return u'(OR: %r %r)' % (self.arg1, self.arg2)


class NotExpr(QueryExpr):
  arity = 1

  def matcher(self, case=False):
    expr = _matcher(self.arg1, case)
    return lambda text: not expr(text)

  def __repr__(self):
    return u'(NOT: %r)' % (self.arg1,)
