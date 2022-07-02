ECHO Building the book...
jb build .
ECHO Done.

ECHO Pushing to gh-page branch...
ghp-import -n -p -f -c tangc.net _build/html
ECHO Done.

PAUSE