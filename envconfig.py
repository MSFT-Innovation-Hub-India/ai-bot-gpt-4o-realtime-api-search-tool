#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
from dotenv import load_dotenv
load_dotenv()

class DefaultConfig:
    """ Bot Configuration """

    az_openai_key=os.getenv("az_openai_key")
    az_openai_baseurl=os.getenv("az_openai_baseurl")
    az_open_ai_endpoint_name=os.getenv("az_open_ai_endpoint_name")
    az_openai_type=os.getenv("az_openai_type")
    az_openai_api_version=os.getenv("az_openai_api_version")
    deployment_name=os.getenv("deployment_name")
    model_name=os.getenv("model_name")
    tavily_key=os.getenv("tavily_key")
