from screen_state import OCRToken, ScreenState, analyze_tokens


def _token(text, left, top, width, height, confidence=90.0):
    return OCRToken(
        text=text,
        left=left,
        top=top,
        width=width,
        height=height,
        confidence=confidence,
    )


def test_detects_countdown_state():
    tokens = [
        _token("Recompensa", 100, 20, 120, 32),
        _token("en", 230, 20, 40, 32),
        _token("6", 280, 20, 30, 32),
        _token("s", 320, 20, 20, 32),
    ]

    state, point = analyze_tokens(tokens, (720, 1280, 3))

    assert state is ScreenState.COUNTDOWN
    assert point is None


def test_detects_continue_arrow_and_point_right():
    token = _token("Continuar", 400, 500, 140, 40)
    state, point = analyze_tokens([token], (1080, 1920, 3))

    assert state is ScreenState.CONTINUE_ARROW
    assert point is not None
    assert point[0] > token.left + token.width
    assert token.top <= point[1] <= token.top + token.height


def test_detects_reward_overlay():
    tokens = [
        _token("Recompensa", 200, 60, 160, 40),
        _token("conseguida", 370, 60, 200, 40),
    ]

    state, point = analyze_tokens(tokens, (1280, 720, 3))

    assert state is ScreenState.REWARD_COMPLETE
    assert point is None


def test_detects_main_reward_button_center():
    tokens = [
        _token("200", 300, 980, 80, 50),
        _token("monedas", 390, 985, 160, 50),
        _token("por", 560, 990, 70, 50),
        _token("ver", 640, 992, 80, 50),
        _token("anuncio", 730, 995, 150, 50),
    ]

    state, point = analyze_tokens(tokens, (1280, 720, 3))

    assert state is ScreenState.MAIN_REWARD_BUTTON
    assert point is not None
    left = min(t.left for t in tokens)
    right = max(t.left + t.width for t in tokens)
    assert left <= point[0] <= right


def test_ignores_low_confidence_tokens():
    tokens = [
        _token("Continuar", 100, 100, 100, 40, confidence=10.0),
        _token("monedas", 100, 800, 120, 40, confidence=10.0),
    ]

    state, point = analyze_tokens(tokens, (1080, 1920, 3))

    assert state is ScreenState.UNKNOWN
    assert point is None


def test_unknown_when_no_tokens():
    state, point = analyze_tokens([], (720, 1280, 3))

    assert state is ScreenState.UNKNOWN
    assert point is None
