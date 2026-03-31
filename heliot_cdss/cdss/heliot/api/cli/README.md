# Heliot CLI Utilities
This module provides command-line tools for managing projects (tenants)
and API keys used to access the Heliot API.

The CLI is intended for development, testing, and administrative
purposes.
------------------------------------------------------------------------

## Requirements
Before using the CLI, make sure that:

-   The database is initialized and migrations have been applied:

        poetry run alembic upgrade head

-   The required environment variables are set in your .env file:

        DATABASE_URL=postgresql://heliot:heliot@localhost:5432/heliot
        HELIOT_API_KEY_PEPPER_V1=<your_api_key_pepper>
------------------------------------------------------------------------

## General Usage
All commands must be executed from the project root using Poetry:

    poetry run python -m cdss.heliot.api.cli.<module> <command> [options]

Available modules: 
- projects → manage tenants 
- api_keys → manage API keys

------------------------------------------------------------------------

## Projects CLI (Examples)
### Create a project

    poetry run python -m cdss.heliot.api.cli.projects create --name "Project1"

### List projects
List all:

    poetry run python -m cdss.heliot.api.cli.projects list

List Only active projects:

    poetry run python -m cdss.heliot.api.cli.projects list --active-only

### Get Project Info
get by ID:

    poetry run python -m cdss.heliot.api.cli.projects get --id 1

Get by name:

    poetry run python -m cdss.heliot.api.cli.projects get-by-name --name "Project1"

### Activate / Deactivate project

    poetry run python -m cdss.heliot.api.cli.projects set-active --id 1 --active false

------------------------------------------------------------------------

## API Keys CLI (Examples)
### Create API key
Create by project name:

    poetry run python -m cdss.heliot.api.cli.api_keys create --project-name "Project1" --env prod --name "key1"

Create by project id:

    poetry run python -m cdss.heliot.api.cli.api_keys create --project-id 1 --env test --name "key1"


### List API keys
List all keys:

    poetry run python -m cdss.heliot.api.cli.api_keys list --project-name "Project1"

List only active keys:

    poetry run python -m cdss.heliot.api.cli.api_keys list --project-id 1 --active-only


### Revoke API key

    poetry run python -m cdss.heliot.api.cli.api_keys revoke --prefix <key_prefix>


### Verify API key (testing helper)

    poetry run python -m cdss.heliot.api.cli.api_keys verify --token <full_token>

------------------------------------------------------------------------

## Notes
API keys follow the format: `hl_sk_<env>_<prefix>_<secret>`
