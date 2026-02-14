import requests
from typing import Optional, Union, List
from urllib.parse import urlencode


class TTSAPIError(Exception):
    """Raised when TTS API returns an error response"""
    pass


# Default reference audio and text paths
# Use relative path assuming API server runs from sovits/ directory
DEFAULT_REF_AUDIO_PATH = "weights/Target.wav"
DEFAULT_PROMPT_TEXT = "文明的命运取决于人类的选择，从不是弱小，而是傲慢。"


class TTS:
    """
    SoVITS TTS API Client
    
    Supports both api.py (v1) and api_v2.py (v2) endpoints.
    Default is v2 API which provides more features and better stability.
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9880,
        api_version: str = "v2",
        timeout: float = 300.0,
        ref_audio_path: str = DEFAULT_REF_AUDIO_PATH,
        prompt_text: str = DEFAULT_PROMPT_TEXT,
        prompt_language: str = "zh"
    ):
        """
        Initialize TTS client
        
        Args:
            host: API server host, default "127.0.0.1"
            port: API server port, default 9880
            api_version: API version, "v1" or "v2", default "v2"
            timeout: Request timeout in seconds, default 300
            ref_audio_path: Default reference audio path
            prompt_text: Default prompt text for the reference audio
            prompt_language: Default language of the prompt text
        """
        self.host = host
        self.port = port
        self.api_version = api_version.lower()
        self.timeout = timeout
        self.default_ref_audio_path = ref_audio_path
        self.default_prompt_text = prompt_text
        self.default_prompt_language = prompt_language
        self.session = requests.Session()
        
        # Build base URL
        self.base_url = f"http://{host}:{port}"
        
        # Set endpoint based on API version
        if self.api_version == "v2":
            self.tts_endpoint = f"{self.base_url}/tts"
        else:
            self.tts_endpoint = f"{self.base_url}/"
    
    def generate(
        self,
        text: str,
        text_language: str = "zh",
        ref_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        prompt_language: Optional[str] = None,
        aux_ref_audio_paths: Optional[List[str]] = None,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        text_split_method: str = "cut5",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        seed: int = -1,
        media_type: str = "wav",
        streaming_mode: Union[bool, int] = False,
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35,
        sample_steps: int = 32,
        super_sampling: bool = False,
        overlap_length: int = 2,
        min_chunk_length: int = 16,
        cut_punc: Optional[str] = None,
    ) -> bytes:
        """
        Generate speech from text using SoVITS API
        
        Args:
            text: Text to be synthesized (required)
            text_language: Language of the text, e.g., "zh", "en", "ja", "ko", "yue"
            ref_audio_path: Path to reference audio file (required for v2, optional for v1)
            prompt_text: Prompt text for the reference audio (optional)
            prompt_language: Language of the prompt text
            aux_ref_audio_paths: Auxiliary reference audio paths for multi-speaker tone fusion (v2 only)
            top_k: Top k sampling, default 15
            top_p: Top p sampling, default 1.0
            temperature: Temperature for sampling, default 1.0
            text_split_method: Text split method (v2 only), default "cut5"
            batch_size: Batch size for inference (v2 only), default 1
            batch_threshold: Threshold for batch splitting (v2 only), default 0.75
            split_bucket: Whether to split the batch into multiple buckets (v2 only), default True
            speed_factor: Control the speed of the synthesized audio, default 1.0
            fragment_interval: Interval between audio fragments, default 0.3
            seed: Random seed for reproducibility, default -1 (random)
            media_type: Output audio format: "wav", "ogg", "aac", "raw", default "wav"
            streaming_mode: Whether to use streaming mode (v2: 0/1/2/3 or bool, v1: bool)
            parallel_infer: Whether to use parallel inference (v2 only), default True
            repetition_penalty: Repetition penalty for T2S model (v2 only), default 1.35
            sample_steps: Number of sampling steps for V3 model (v2 only), default 32
            super_sampling: Whether to use super-sampling for V3 model (v2 only), default False
            overlap_length: Overlap length for streaming mode (v2 only), default 2
            min_chunk_length: Minimum chunk length for streaming mode (v2 only), default 16
            cut_punc: Punctuation for text cutting (v1 only), e.g., "，。"
            
        Returns:
            bytes: Generated audio data
            
        Raises:
            requests.RequestException: When API request fails
            ValueError: When required parameters are missing
        """
        if not text:
            raise ValueError("text is required")
        
        # Use default values if not specified
        if ref_audio_path is None:
            ref_audio_path = self.default_ref_audio_path
        if prompt_text is None:
            prompt_text = self.default_prompt_text
        if prompt_language is None:
            prompt_language = self.default_prompt_language
        
        # Build request parameters based on API version
        if self.api_version == "v2":
            if not ref_audio_path:
                raise ValueError("ref_audio_path is required for API v2")
            
            params = {
                "text": text,
                "text_lang": text_language.lower(),
                "ref_audio_path": ref_audio_path,
                "prompt_lang": prompt_language.lower() if prompt_language else text_language.lower(),
                "prompt_text": prompt_text or "",
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "text_split_method": text_split_method,
                "batch_size": batch_size,
                "batch_threshold": batch_threshold,
                "split_bucket": split_bucket,
                "speed_factor": speed_factor,
                "fragment_interval": fragment_interval,
                "seed": seed,
                "media_type": media_type,
                "streaming_mode": streaming_mode,
                "parallel_infer": parallel_infer,
                "repetition_penalty": repetition_penalty,
                "sample_steps": sample_steps,
                "super_sampling": super_sampling,
                "overlap_length": overlap_length,
                "min_chunk_length": min_chunk_length,
            }
            
            # Add optional parameters
            if aux_ref_audio_paths:
                params["aux_ref_audio_paths"] = aux_ref_audio_paths
                
        else:  # v1 API
            params = {
                "text": text,
                "text_language": text_language.lower(),
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "speed": speed_factor,
                "media_type": media_type,
            }
            
            # Add reference audio parameters if provided
            if ref_audio_path:
                params["refer_wav_path"] = ref_audio_path
            if prompt_text:
                params["prompt_text"] = prompt_text
            if prompt_language:
                params["prompt_language"] = prompt_language.lower()
            if cut_punc:
                params["cut_punc"] = cut_punc
                
            # v1 streaming mode (boolean)
            if isinstance(streaming_mode, bool):
                params["streaming_mode"] = streaming_mode
                
            # v1 inp_refs (auxiliary reference audios)
            if aux_ref_audio_paths:
                params["inp_refs"] = aux_ref_audio_paths
        
        # Build URL for debugging
        full_url = self._build_url(params)
        
        # Send request
        response = self.session.get(
            self.tts_endpoint,
            params=params,
            timeout=self.timeout,
            stream=bool(streaming_mode)
        )
        
        # Check response
        if response.status_code == 400:
            try:
                error_data = response.json()
                raise TTSAPIError(f"API Error: {error_data}, URL: {full_url}")
            except ValueError:
                raise TTSAPIError(f"API Error (400): {response.text}, URL: {full_url}")
        
        response.raise_for_status()
        
        return response.content
    
    def generate_post(
        self,
        text: str,
        text_language: str = "zh",
        ref_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        prompt_language: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Generate speech using POST request (recommended for long text)
        
        Args:
            text: Text to be synthesized (required)
            text_language: Language of the text
            ref_audio_path: Path to reference audio file (uses default if not specified)
            prompt_text: Prompt text for the reference audio (uses default if not specified)
            prompt_language: Language of the prompt text (uses default if not specified)
            **kwargs: Additional parameters (same as generate method)
            
        Returns:
            bytes: Generated audio data
        """
        if not text:
            raise ValueError("text is required")
        
        # Use default values if not specified
        if ref_audio_path is None:
            ref_audio_path = self.default_ref_audio_path
        if prompt_text is None:
            prompt_text = self.default_prompt_text
        if prompt_language is None:
            prompt_language = self.default_prompt_language
        
        if self.api_version == "v2":
            if not ref_audio_path:
                raise ValueError("ref_audio_path is required for API v2")
            
            data = {
                "text": text,
                "text_lang": text_language.lower(),
                "ref_audio_path": ref_audio_path,
                "prompt_lang": prompt_language.lower() if prompt_language else text_language.lower(),
                "prompt_text": prompt_text or "",
                "top_k": kwargs.get("top_k", 15),
                "top_p": kwargs.get("top_p", 1.0),
                "temperature": kwargs.get("temperature", 1.0),
                "text_split_method": kwargs.get("text_split_method", "cut5"),
                "batch_size": kwargs.get("batch_size", 1),
                "batch_threshold": kwargs.get("batch_threshold", 0.75),
                "split_bucket": kwargs.get("split_bucket", True),
                "speed_factor": kwargs.get("speed_factor", 1.0),
                "fragment_interval": kwargs.get("fragment_interval", 0.3),
                "seed": kwargs.get("seed", -1),
                "media_type": kwargs.get("media_type", "wav"),
                "streaming_mode": kwargs.get("streaming_mode", False),
                "parallel_infer": kwargs.get("parallel_infer", True),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.35),
                "sample_steps": kwargs.get("sample_steps", 32),
                "super_sampling": kwargs.get("super_sampling", False),
                "overlap_length": kwargs.get("overlap_length", 2),
                "min_chunk_length": kwargs.get("min_chunk_length", 16),
            }
            
            aux_ref_audio_paths = kwargs.get("aux_ref_audio_paths")
            if aux_ref_audio_paths:
                data["aux_ref_audio_paths"] = aux_ref_audio_paths
                
        else:  # v1 API
            data = {
                "text": text,
                "text_language": text_language.lower(),
                "top_k": kwargs.get("top_k", 15),
                "top_p": kwargs.get("top_p", 1.0),
                "temperature": kwargs.get("temperature", 1.0),
                "speed": kwargs.get("speed_factor", 1.0),
            }
            
            if ref_audio_path:
                data["refer_wav_path"] = ref_audio_path
            if prompt_text:
                data["prompt_text"] = prompt_text
            if prompt_language:
                data["prompt_language"] = prompt_language.lower()
        
        response = self.session.post(
            self.tts_endpoint,
            json=data,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.content
    
    def change_reference(
        self,
        refer_wav_path: str,
        prompt_text: str,
        prompt_language: str
    ) -> dict:
        """
        Change default reference audio (v1 API only)
        
        Args:
            refer_wav_path: Path to reference audio file
            prompt_text: Prompt text for the reference audio
            prompt_language: Language of the prompt text
            
        Returns:
            dict: API response
        """
        if self.api_version == "v2":
            raise NotImplementedError("change_reference is not available in API v2. Use ref_audio_path in generate() instead.")
        
        endpoint = f"{self.base_url}/change_refer"
        params = {
            "refer_wav_path": refer_wav_path,
            "prompt_text": prompt_text,
            "prompt_language": prompt_language.lower(),
        }
        
        response = self.session.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def control(self, command: str) -> None:
        """
        Send control command to the API server
        
        Args:
            command: Command to send, "restart" or "exit"
        """
        if command not in ("restart", "exit"):
            raise ValueError("command must be 'restart' or 'exit'")
        
        endpoint = f"{self.base_url}/control"
        params = {"command": command}
        
        self.session.get(endpoint, params=params, timeout=30)
    
    def set_gpt_weights(self, weights_path: str) -> dict:
        """
        Switch GPT model weights (v2 API)
        
        Args:
            weights_path: Path to GPT weights file
            
        Returns:
            dict: API response
        """
        if self.api_version != "v2":
            raise NotImplementedError("set_gpt_weights is only available in API v2")
        
        endpoint = f"{self.base_url}/set_gpt_weights"
        params = {"weights_path": weights_path}
        
        response = self.session.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def set_sovits_weights(self, weights_path: str) -> dict:
        """
        Switch SoVITS model weights (v2 API)
        
        Args:
            weights_path: Path to SoVITS weights file
            
        Returns:
            dict: API response
        """
        if self.api_version != "v2":
            raise NotImplementedError("set_sovits_weights is only available in API v2")
        
        endpoint = f"{self.base_url}/set_sovits_weights"
        params = {"weights_path": weights_path}
        
        response = self.session.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def set_reference_audio(self, refer_audio_path: str) -> dict:
        """
        Set reference audio (v2 API)
        
        Args:
            refer_audio_path: Path to reference audio file
            
        Returns:
            dict: API response
        """
        if self.api_version != "v2":
            raise NotImplementedError("set_reference_audio is only available in API v2")
        
        endpoint = f"{self.base_url}/set_refer_audio"
        params = {"refer_audio_path": refer_audio_path}
        
        response = self.session.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def check_api(self) -> dict:
        """
        Check if API server is running and get basic info
        
        Returns:
            dict: API info if available
        """
        try:
            response = self.session.get(self.base_url, timeout=10)
            return {
                "status": "running" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "api_version": self.api_version,
                "url": self.base_url
            }
        except requests.RequestException as e:
            return {
                "status": "unreachable",
                "error": str(e),
                "api_version": self.api_version,
                "url": self.base_url
            }
    
    def validate_files(self) -> dict:
        """
        Validate that default reference files exist locally
        
        Returns:
            dict: Validation results
        """
        import os
        results = {
            "ref_audio_path": {
                "path": self.default_ref_audio_path,
                "exists": os.path.exists(self.default_ref_audio_path),
                "readable": os.access(self.default_ref_audio_path, os.R_OK) if os.path.exists(self.default_ref_audio_path) else False
            },
            "prompt_text": self.default_prompt_text
        }
        return results
    
    def _build_url(self, params: dict) -> str:
        """Build clean URL for debugging (without encoding)"""
        # Filter out None values and empty lists
        clean_params = {k: v for k, v in params.items() if v is not None and v != []}
        # Build query string without encoding
        query_parts = []
        for k, v in clean_params.items():
            if isinstance(v, bool):
                v = str(v).lower()
            elif isinstance(v, (int, float)):
                v = str(v)
            query_parts.append(f"{k}={v}")
        query_string = "&".join(query_parts)
        return f"{self.tts_endpoint}?{query_string}"


# Convenience function for quick TTS generation
def tts(
    text: str,
    text_language: str = "zh",
    ref_audio_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    prompt_language: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 9880,
    api_version: str = "v2",
    **kwargs
) -> bytes:
    """
    Quick TTS generation function
    
    Args:
        text: Text to be synthesized
        text_language: Language of the text
        ref_audio_path: Path to reference audio file (uses default from TTS class if not specified)
        prompt_text: Prompt text for the reference audio (uses default from TTS class if not specified)
        prompt_language: Language of the prompt text (uses default from TTS class if not specified)
        host: API server host
        port: API server port
        api_version: API version, "v1" or "v2"
        **kwargs: Additional parameters passed to TTS.generate()
        
    Returns:
        bytes: Generated audio data
    """
    client = TTS(host=host, port=port, api_version=api_version)
    return client.generate(
        text=text,
        text_language=text_language,
        ref_audio_path=ref_audio_path,
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage with debugging
    
    # NOTE: Adjust ref_audio_path based on where your API server is running from:
    # - If API runs from sovits/: use "weights/Target.wav" (relative)
    # - If API runs from project root: use "sovits/weights/Target.wav"
    # - If API runs elsewhere: use absolute path "/home/.../Target.wav"
    
    client = TTS(
        host="127.0.0.1",
        port=9880,
        api_version="v2",
        # Adjust this path based on your API server's working directory:
        ref_audio_path="weights/Target.wav",  # Relative to API server's CWD
        prompt_text="文明的命运取决于人类的选择，从不是弱小，而是傲慢。",
        prompt_language="zh"
    )
    
    # Check API status
    print("API Status:", client.check_api())
    
    # Validate local files
    print("File Validation:", client.validate_files())
    
    # Show the URL that will be used
    print(f"\nDefault ref_audio_path: {client.default_ref_audio_path}")
    print(f"Default prompt_text: {client.default_prompt_text}")
    
    # Try to generate
    try:
        audio = client.generate(
            text="test",
            text_language="zh"
        )
        with open("output.wav", "wb") as f:
            f.write(audio)
        print("Success! Audio saved to output.wav")
    except TTSAPIError as e:
        print(f"\nTTS API Error: {e}")
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
