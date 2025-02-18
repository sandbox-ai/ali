import { Component, Input, OnInit, OnDestroy, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-text-stream',
  template: `
    <div class="text-container">
      <p class="typing" [innerHTML]="formattedText"></p>
    </div>
  `,
  styles: [`
    .text-container {
      width: min(970px, 90%);
      margin: 0 auto;
    }
    :host ::ng-deep .typing {
      font-size: 16px;
      font-family: 'JustusPro-Thin', sans-serif;
      line-height: 26px;
      text-align: justify;
      white-space: pre-wrap;
    }
  `]
})
export class TextStreamComponent implements OnInit, OnDestroy {
  @Input() text: string = '';
  @Output() completed = new EventEmitter<void>();
  displayText: string = '';
  formattedText: string = '';
  private animationFrameId: number | null = null;
  private startTime: number = 0;
  private readonly CHARS_PER_FRAME = 4; // Adjust this for speed

  ngOnInit() {
    this.startTime = performance.now();
    this.animateText();
  }

  ngOnDestroy() {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }
  }

  private animateText = () => {
    const currentTime = performance.now();
    const elapsed = currentTime - this.startTime;
    
    // Calculate how many characters should be shown by now
    const targetLength = Math.floor(elapsed / 16.67 * this.CHARS_PER_FRAME);
    
    if (targetLength < this.text.length) {
      this.displayText = this.text.slice(0, targetLength);
      // Convert newlines to <br> tags and preserve whitespace
      this.formattedText = this.displayText.replace(/\n/g, '<br>');
      this.animationFrameId = requestAnimationFrame(this.animateText);
    } else {
      this.displayText = this.text;
      this.formattedText = this.text.replace(/\n/g, '<br>');
      this.animationFrameId = null;
      this.completed.emit();
    }
  };
} 