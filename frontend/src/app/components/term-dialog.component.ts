import { Component, Input } from '@angular/core';
import { NbDialogRef } from '@nebular/theme';

@Component({
  template: `
    <nb-card class="dialog-card">
      <!-- <nb-card-header>Términos y condiciones</nb-card-header> -->
      <nb-card-body>
        {{ message }}
      </nb-card-body>
      <!--
        <nb-card-footer>
          <button nbButton status="primary" (click)="dismiss()">Dismiss Dialog</button>
        </nb-card-footer>
      -->
    </nb-card>
  `,
})
export class TermDialogComponent {
  @Input() message: string = "";

  constructor(protected ref: NbDialogRef<TermDialogComponent>) {
  }

  dismiss() {
    this.ref.close();
  }
}
