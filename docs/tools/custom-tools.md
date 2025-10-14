# Custom Tools

This section provides examples of custom tools that can be integrated into the AI Assistant System.

## Calculator Tool

A simple calculator tool for basic arithmetic operations.

```python
from app.core.tools.base import BaseTool, ToolResult
from pydantic import BaseModel
from typing import Union
import operator

class CalculatorParameters(BaseModel):
    operation: str
    a: Union[int, float]
    b: Union[int, float]

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform basic arithmetic operations"
    parameters_schema = CalculatorParameters
    
    operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
        "power": operator.pow,
    }
    
    async def execute(self, parameters: CalculatorParameters) -> ToolResult:
        if parameters.operation not in self.operations:
            return ToolResult(
                success=False,
                error=f"Unsupported operation: {parameters.operation}",
                error_code="INVALID_OPERATION"
            )
        
        try:
            result = self.operations[parameters.operation](parameters.a, parameters.b)
            return ToolResult(
                success=True,
                data={"result": result, "operation": parameters.operation}
            )
        except ZeroDivisionError:
            return ToolResult(
                success=False,
                error="Division by zero",
                error_code="DIVISION_BY_ZERO"
            )
```

## URL Shortener Tool

A tool for shortening URLs using an external service.

```python
import httpx
from app.core.tools.base import BaseTool, ToolResult
from pydantic import BaseModel
import os

class URLShortenerParameters(BaseModel):
    url: str

class URLShortenerTool(BaseTool):
    name = "url_shortener"
    description = "Shorten URLs using a URL shortening service"
    parameters_schema = URLShortenerParameters
    
    def __init__(self):
        self.api_key = os.getenv("URL_SHORTENER_API_KEY")
        self.base_url = "https://api.shorten.io/v1"
    
    async def execute(self, parameters: URLShortenerParameters) -> ToolResult:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"url": parameters.url}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/shorten",
                headers=headers,
                json=data
            )
            
        if response.status_code == 200:
            result = response.json()
            return ToolResult(
                success=True,
                data={"short_url": result["short_url"], "original_url": parameters.url}
            )
        else:
            return ToolResult(
                success=False,
                error=f"API request failed with status {response.status_code}",
                error_code="API_ERROR"
            )
```

## Image Generator Tool

A tool for generating images using an AI image generation service.

```python
import httpx
from app.core.tools.base import BaseTool, ToolResult
from pydantic import BaseModel
from typing import Optional
import os

class ImageGeneratorParameters(BaseModel):
    prompt: str
    size: Optional[str] = "512x512"
    style: Optional[str] = "realistic"

class ImageGeneratorTool(BaseTool):
    name = "image_generator"
    description = "Generate images from text prompts"
    parameters_schema = ImageGeneratorParameters
    
    def __init__(self):
        self.api_key = os.getenv("IMAGE_GENERATOR_API_KEY")
        self.base_url = "https://api.imagegen.io/v1"
    
    async def execute(self, parameters: ImageGeneratorParameters) -> ToolResult:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "prompt": parameters.prompt,
            "size": parameters.size,
            "style": parameters.style
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/generate",
                headers=headers,
                json=data
            )
            
        if response.status_code == 200:
            result = response.json()
            return ToolResult(
                success=True,
                data={
                    "image_url": result["image_url"],
                    "prompt": parameters.prompt,
                    "size": parameters.size
                }
            )
        else:
            return ToolResult(
                success=False,
                error=f"Image generation failed: {response.text}",
                error_code="GENERATION_ERROR"
            )
```

## Email Sender Tool

A tool for sending emails via SMTP.

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.tools.base import BaseTool, ToolResult
from pydantic import BaseModel
from typing import List
import os

class EmailSenderParameters(BaseModel):
    to: List[str]
    subject: str
    body: str
    html_body: Optional[str] = None

class EmailSenderTool(BaseTool):
    name = "email_sender"
    description = "Send emails via SMTP"
    parameters_schema = EmailSenderParameters
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.username = os.getenv("SMTP_USERNAME")
        self.password = os.getenv("SMTP_PASSWORD")
    
    async def execute(self, parameters: EmailSenderParameters) -> ToolResult:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = parameters.subject
            msg['From'] = self.username
            msg['To'] = ', '.join(parameters.to)
            
            # Add plain text part
            msg.attach(MIMEText(parameters.body, 'plain'))
            
            # Add HTML part if provided
            if parameters.html_body:
                msg.attach(MIMEText(parameters.html_body, 'html'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.username, parameters.to, text)
            server.quit()
            
            return ToolResult(
                success=True,
                data={
                    "message": "Email sent successfully",
                    "recipients": parameters.to,
                    "subject": parameters.subject
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to send email: {str(e)}",
                error_code="EMAIL_ERROR"
            )
```

## File Converter Tool

A tool for converting files between different formats.

```python
import subprocess
from pathlib import Path
from app.core.tools.base import BaseTool, ToolResult
from pydantic import BaseModel
import tempfile
import os

class FileConverterParameters(BaseModel):
    input_file: str
    output_format: str
    output_file: Optional[str] = None

class FileConverterTool(BaseTool):
    name = "file_converter"
    description = "Convert files between different formats"
    parameters_schema = FileConverterParameters
    
    async def execute(self, parameters: FileConverterParameters) -> ToolResult:
        input_path = Path(parameters.input_file)
        
        if not input_path.exists():
            return ToolResult(
                success=False,
                error="Input file not found",
                error_code="FILE_NOT_FOUND"
            )
        
        # Generate output filename if not provided
        if not parameters.output_file:
            output_file = f"{input_path.stem}.{parameters.output_format}"
        else:
            output_file = parameters.output_file
        
        output_path = Path(output_file)
        
        try:
            # Example: Convert PDF to text using pdftotext
            if input_path.suffix.lower() == '.pdf' and parameters.output_format == 'txt':
                result = subprocess.run(
                    ['pdftotext', str(input_path), str(output_path)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    return ToolResult(
                        success=True,
                        data={
                            "output_file": str(output_path),
                            "input_file": str(input_path),
                            "format": parameters.output_format
                        }
                    )
                else:
                    return ToolResult(
                        success=False,
                        error=f"Conversion failed: {result.stderr}",
                        error_code="CONVERSION_ERROR"
                    )
            
            # Add more conversion logic as needed
            else:
                return ToolResult(
                    success=False,
                    error="Unsupported conversion",
                    error_code="UNSUPPORTED_CONVERSION"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Conversion error: {str(e)}",
                error_code="CONVERSION_ERROR"
            )
```

## Using Custom Tools

To use these custom tools:

1. Save the tool code in the `app/core/tools/custom/` directory
2. Import and register the tools in your application
3. Configure any required environment variables
4. Test the tools before using them in production

## Security Considerations

When creating custom tools:

1. **Validate Inputs**: Always validate user inputs
2. **Sanitize Data**: Sanitize data before processing
3. **Limit Permissions**: Run tools with minimal required permissions
4. **Handle Errors**: Implement proper error handling
5. **Rate Limiting**: Consider implementing rate limiting for external API calls
6. **Logging**: Log tool usage for auditing purposes