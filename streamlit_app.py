"""The web page's entry point: ``streamlit run streamlit_app.py`` (also what Streamlit Community Cloud runs).

The page itself is ``ratesplot.web``; ``ratesplot.py`` is the
command line and the desktop window.
"""

from ratesplot.web import main

main()
