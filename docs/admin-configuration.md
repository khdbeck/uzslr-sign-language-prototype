# Admin access

Admin login is disabled unless the server's `ADMIN_PASSWORD` environment variable
is set to a non-empty value. Configure a unique password through your hosting
provider's secret/environment settings; do not commit it to Git.

Earlier versions inherited a hard-coded demonstration password from the upstream
project. Treat that value as public and never reuse it. Updating the code does not
remove it from Git history. Existing deployments must be redeployed to disable it.

The normal sign-recognition interface does not require admin access. Restart the
server after changing the password; existing in-memory admin tokens are cleared
on restart.
