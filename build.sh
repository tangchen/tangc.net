echo Building the book...
jb build .
echo Done.

echo Pushing to gh-page branch...
ghp-import -n -p -f -c tangc.net _build/html
echo Done.