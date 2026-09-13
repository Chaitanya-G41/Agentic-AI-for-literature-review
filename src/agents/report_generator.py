from pydantic import BaseModel, Field
from typing import List
import os
import re
import json
import google.generativeai as genai

class LiteratureReviewReport:
  
