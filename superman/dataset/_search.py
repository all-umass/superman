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

  # AND expressions bind tightest, then OR, then adjacency (implicit AND)
  for lexeme, expr_type in [(u'AND', AndExpr), (u'OR', OrExpr)]:
    while True:
      idxs = [i for i, x in enumerate(tokens)
              if not isinstance(x, QueryExpr) and x.upper() == lexeme]
      if not idxs:
        break
      idx = idxs[0]
      lhs = tokens[idx-1]
      rhs = tokens[idx+1]
      tokens = tokens[:idx-1] + [expr_type(lhs, rhs)] + tokens[idx+2:]

  # AND together any remaining roots
  while len(tokens) > 1:
    rhs = tokens.pop()
    lhs = tokens.pop()
    tokens.append(AndExpr(lhs, rhs))

  # get a matcher function from the root
  root, = tokens
  return _matcher(root, case_sensitive)


def _matcher(token, case_sensitive):
  if isinstance(token, QueryExpr):
    return token.matcher(case=case_sensitive)
  elif case_sensitive:
    return lambda text: token in text
  else:
    low = token.lower()
    return lambda text: low in text.lower()


class QueryExpr(object):
  def __init__(self, lhs, rhs):
    self.lhs = lhs
    self.rhs = rhs


class AndExpr(QueryExpr):
  def matcher(self, case=False):
    lhs = _matcher(self.lhs, case)
    rhs = _matcher(self.rhs, case)
    return lambda text: lhs(text) and rhs(text)

  def __repr__(self):
    return u'(AND: %r %r)' % (self.lhs, self.rhs)


class OrExpr(QueryExpr):
  def matcher(self, case=False):
    lhs = _matcher(self.lhs, case)
    rhs = _matcher(self.rhs, case)
    return lambda text: lhs(text) or rhs(text)

  def __repr__(self):
    return u'(OR: %r %r)' % (self.lhs, self.rhs)
